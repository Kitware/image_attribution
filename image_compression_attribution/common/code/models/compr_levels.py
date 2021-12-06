"""Model for image attribution to news sources via compression levels.

Uses imagemagick's identify to estimate the compression quality level of
an image.  Uses a different library, python-magic, to read the MIME
type of an image (to detect if it is really an image).
"""

import subprocess
import logging
import numpy as np
import pandas as pd

import errno
import os

import magic # note: python-magic, not file-magic

ALLOWED_COMPRESSION_LEVELS = np.arange(-1,101, dtype=int)

def get_mime(filepath):
  """Infers file mime type"""
  #use python-magic to read the mime type of a file. This lets us detect
  #if an image is an image file, which starts with "image/", or not.
  #E.g. sometimes downloads fail and files that should be images are
  #really html files with download error codes. This lets us detect those.
  mime = magic.Magic(mime=True)
  return mime.from_file(filepath)

def read_compression_level(img_filepath, filter_on_mime=True):
  """Read compression level of image file. 
  
  Calls imagemagick via subprocess and parses out compression level(s).
  If multiple levels present (e.g. for gif file), return the int() of the mean.
  
  Args:
    img_filepath: str, path to image file to load.
    filter_on_mime: only process a file if python-magic detects its mime
      type starts with 'image', e.g. 'image/jpeg'
    
  Returns:
    single compression level, -1 for error. If multiple levels, returns
    the mean of the levels converted to an integer.

  Raises: 
    FileNotFoundError: if image file (img_filepath) does not exist
    ValueError: raised if filter_on_mime is True and img_filepath's mime type 
      does not start with "image",
  """  
  global ALLOWED_COMPRESSION_LEVELS

  logging.debug("in read_compression_level()")
  if os.path.isfile(img_filepath):
    if filter_on_mime:
      mime_string = get_mime(img_filepath)
      if not mime_string.startswith('image'): #not an image file, so don't process it
        logging.debug("bad mime type {}".format(mime_string))
        raise ValueError("Unsupported file format: mime {} detected for file {}".format(
          mime_string, img_filepath))

    #use imagemagick to read the compression level of the image file
    logging.debug("About to call imagemagick's identify from subprocess")
    response_string = subprocess.getoutput(
      "identify -quiet -format '%Q ' {} 2>/dev/null".format(img_filepath))   # The 2>/dev/null hides stderr warnings, see https://stackoverflow.com/a/12759182
    logging.debug("response string: {}".format(response_string))
  else:
    raise FileNotFoundError(
      errno.ENOENT, os.strerror(errno.ENOENT), img_filepath)
  logging.debug("Completed call to imagamagick's identify from subprocess")
  
  if response_string.startswith("identify:"):
    #Error reading the file, return -1 for error
    logging.warn("Error '{}' when loading file {}".format(response_string, \
      img_filepath))
    return -1
  else:
    #Some files like gif have multiple resolutions with multiple compressions
    #so parse out potentially multiple compression levels
    comp_levels = [int(word) for word in response_string.split() if \
      word.isdigit()]
    if len(comp_levels) == 1: 
      comp_level = int(comp_levels[0])
    elif len(comp_levels) > 1: #multiple levels, return the mean
      logging.debug("{} compression levels found in image {}".format(
        comp_levels, img_filepath))
      comp_level = np.array(comp_levels).mean().astype(int)
    else: #len ==0 --> unexpected error
      logging.warn("Unexpected output '{}' for file {}".format(
        response_string, img_filepath))
      comp_level = -1
    #detect invalid values
    if comp_level not in ALLOWED_COMPRESSION_LEVELS:
      logging.warn("Invalid compression level {} computed for file {}".format(
        comp_level, img_filepath))
      comp_level = -1
    return comp_level

#==============================================================================

class attribution_compression_levels:
  """Class to build a predictive model to verify an image's source.
  
  Implements a categorical Naive Bayes classifier, with compression quality 
  levels as the feature.  Predictions are calibrated log likelihood ratios 
  (LLRs), but you can just treat them as scores for ROC curves.
  """
  def __init__(self):
    global ALLOWED_COMPRESSION_LEVELS
    self.compression_levels = ALLOWED_COMPRESSION_LEVELS
    self.num_levels = 101
    self.smooth_const = 1.0     #Laplace smoothing constant (for Naive Bayes model)
    self.known_sources = []
    self.model = {}
    self.full_train_size = 0
  
  def fit(self, df_train):
    """Fit a model to predict is an image does not come from a claimed source.
       df_train is a training subset of a summarized dataset, as produced
       by summarize_quant_matrices.py"""
    self.known_sources = sorted(list(df_train['source'].unique()))
    self.full_train_size = len(df_train['source'])
    
    for source in self.known_sources:
      #Prepare data:
      #split train data into samples from the correct source, or from all other sources
      X_train_fromsource = df_train.loc[df_train['source']==source, 'compression']
      X_train_notfromsource = df_train.loc[df_train['source']!=source, 'compression']

      compression_counts_fromsource = X_train_fromsource.value_counts() #how many samples (from the source) have each compression level?
      num_fromsource = compression_counts_fromsource.sum() #count total number of samples (from the source)

      compression_counts_notfromsource = X_train_notfromsource.value_counts() #how many samples (not from the source) have each compression level?
      num_notfromsource = compression_counts_notfromsource.sum() #count total number of samples (not from the source)

      #---------------------------------------------------------------
      #From training data, compute model to predict the probability
      #of encountering a compression level l from a source

      prob_fromsource = {}
      for l in self.compression_levels:
        count = self.smooth_const
        if l in compression_counts_fromsource:
          count += compression_counts_fromsource[l]
        prob_fromsource[l] = count/(num_fromsource + self.smooth_const*self.num_levels)

      prob_notfromsource = {}
      for l in self.compression_levels:
        count = self.smooth_const
        if l in compression_counts_notfromsource:
          count += compression_counts_notfromsource[l]
        prob_notfromsource[l] = count/(num_notfromsource + self.smooth_const*self.num_levels)
      
      self.model[source] = {}
      self.model[source]['prob_fromsource'] = prob_fromsource
      self.model[source]['prob_notfromsource'] = prob_notfromsource
      
  def _predict_single(self, compr_level, claimed_source):
    """Predict if a single compression level is inconsistent with a claimed source"""
    if compr_level not in self.compression_levels:
      raise ValueError("{} is not a recognized compression level".format(compr_level))
    
    if claimed_source not in self.known_sources:
      logging.warn("unrecognized source {}".format(claimed_source))
      prob_fromsource = 1.0/self.full_train_size
      prob_notfromsource = 1.0/self.full_train_size
      unrecognized_source = True
    else:
      prob_fromsource = self.model[claimed_source]['prob_fromsource'][compr_level]
      prob_notfromsource = self.model[claimed_source]['prob_notfromsource'][compr_level]
      unrecognized_source = False
    LLR_isfake = np.log10(prob_notfromsource/prob_fromsource)
    return LLR_isfake, prob_fromsource, prob_notfromsource, unrecognized_source
    
  def _predict_multiple(self, compr_levels, claimed_sources):
    """predict for an array of compression levels if they are inconsistent with the claimed source"""
    if len(compr_levels) != len(claimed_sources):
      raise ValueError("Args compr_levels and claimed_sources have differing lengths")
    
    LLRs_isfake = []
    probs_fromsource = []
    probs_notfromsource = []
    unrecognized_sources = []
    for i in range(len(compr_levels)):
      llr, pfs, pnfs, us = self._predict_single(compr_levels[i], claimed_sources[i])
      LLRs_isfake.append(llr)
      probs_fromsource.append(pfs)
      probs_notfromsource.append(pnfs)
      unrecognized_sources.append(us)
    return LLRs_isfake, probs_fromsource, probs_notfromsource, unrecognized_sources
  
  def predict(self, compr_level, claimed_source):
    """Predict is one or more compression levels are inconsistent with a source.
       Flexibly can accept single compression levels or lists of compression 
       levels."""
    cl_arraylike = isinstance(compr_level,(list,pd.core.series.Series,np.ndarray))
    cs_arraylike = isinstance(claimed_source,(list,pd.core.series.Series,np.ndarray))
    if cl_arraylike != cs_arraylike:
      raise ValueError("Arguments must both be scalars or both array-like")
      
    if cl_arraylike:
      return self._predict_multiple(compr_level, claimed_source)
    else:
      return self._predict_single(compr_level, claimed_source)

  def predict_image(self, img_filepath, claimed_source, filter_on_mime=True):
    """Predicts if an image is inconsistent with a claimed source.  If
    The returned LLR > 0, the image is inconsistent. If the LLR < 0, the
    image is consistent with the claimed source.
    """
    compr_level = read_compression_level(img_filepath, filter_on_mime)
    return self.predict(compr_level, claimed_source)
