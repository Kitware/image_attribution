"""Model for image attribution to news sources including quantization matrices.

In this improved model, we include the file mime type, the image compression 
quality level, AND (for jpeg's only) the quantization matrices in the 
features. Note that the quantization matrices allow us to differentiate
different jpeg encoders even when the compression quality levels of two
files are identical.  This improves attribution performance on images with
common compression levels.  The combination of file mime type, compression
level, and quantization matrix are collectively called 'compression settings'.
"""

import os
from typing import Dict, Any
import logging
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB
from PIL import Image


import hashlib
import json

from image_compression_attribution.common.code.models import compr_levels

#Name to use for quantization features when there's an unseen compression setting
#We explicitly include an unseen category, since values which have not been
#seen before in training may occur at testing or in the wild, so the code must 
#gracefully handle those cases.
COMPR_NAME_UNKNOWN = "U"

class UnknownSourceError(ValueError):
  """Raised if a source is not recognized"""
  pass


def read_quantization_matrices(img_filepath):
  """Reads a jpg's quantization matrices, as a dict
  
  Args:
    img_filepath: str, path to image to read
    
  Returns:
    A dict, with integer keys (to specify a quantization matrix) and values
    as 1D lists of integer quantization matrix coefficients.
    
  Raises:
    ValueError: if the image is not a jpeg, according to the file mimetype
    FileNotFoundError: if the file does not exist
  """
  
  logging.debug("in read_quantization_matrices()")
  if os.path.isfile(img_filepath):
    mime_string = compr_levels.get_mime(img_filepath)
    if not mime_string.startswith('image/jpeg'): #not a jpeg file, so don't process it
      #logging.debug("bad mime type {}".format(mime_string))
      raise ValueError("Unsupported file format: mime {} detected for file {}".format(
        mime_string, img_filepath))
  
    logging.debug("About to call imagemagick's identify from subprocess")
    img = Image.open(img_filepath)
    q_tmp = img.quantization
    q_dict = {}
    for key,value in q_tmp.items():
      q_dict[key] = list(value)
    return q_dict
    
    logging.debug("Got quantization matrices: {}".format(q_dict))
  else:
    raise FileNotFoundError(
      errno.ENOENT, os.strerror(errno.ENOENT), img_filepath)
    
#https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
def dict_hash(dictionary: Dict[str, Any]) -> str:
  """MD5 hash of a dictionary."""
  dhash = hashlib.md5()
  # We need to sort arguments so {'a': 1, 'b': 2} is
  # the same as {'b': 2, 'a': 1}
  encoded = json.dumps(dictionary, sort_keys=True).encode()
  dhash.update(encoded)
  return dhash.hexdigest()

def get_filetype_name(mime, file_ext, get_filetype_from_mime):
  """Return a filetype as a string, like 'png'
  
  Args:
    mime: str, mime representation from a file, like "image/png".
    file_ext: str, with file extension, like ".png"
    get_filetype_from_mime: bool, if True, infer filetype (e.g "png") from the
      mime type (e.g. "image/png"). If False, infer the filename from the file 
      extension (e.g ".png")
  Returns:
    str with file type, e.g. "png"
  """
  ext_short = ''
  if not get_filetype_from_mime: #use file extension to get filetype
    if len(ext_raw) > 0:
      ext_short = file_ext[1:].lower()     #e.g. ".jpeg" --> "jpeg"
  else:  #use detected mime type to infer filetype
    filetype, fileformat = mime.split("/")  #e.g. "image/jpeg" --> "image", "jpeg"
    if filetype == 'image':
      ext_short = fileformat    #e.g. "image/jpeg" --> "jpeg"
      
  return ext_short

def get_img_features(img_filepath):
  """Computes quantization-related features from an image
  
  Args:
    img_filepath: str, filepath to image

  Returns:
    compression_level: int, compression quality factor of image
    mime: str, mime type of image (like 'image/jpeg')
    file_extension: str, file extension from filename, e.g. '.jpeg'
    q_hash: str, hash representation of the quantization matrix, or '' if error
    q_matrix: dict containing quantization matrices from image, or None if error
  """
  
  file_extension = os.path.splitext(img_filepath)[1]
  mime = compr_levels.get_mime(img_filepath)

  compression_level = compr_levels.read_compression_level(img_filepath)
  
  if mime == "image/jpeg":
    q_matrix = read_quantization_matrices(img_filepath)
    q_hash = dict_hash(q_matrix)
  else:
    q_matrix = None
    q_hash = ""
          
  return compression_level, mime, file_extension, q_hash, q_matrix

def make_feature_name(compression_level, mime, file_extension, q_hash,
                      get_filetype_from_mime=True, num_hash_digits=5):
  """Create a string name describing compression features of an image.
  
  Args:
    compression level: int, estimated compression level of image
    mime: str, mime type of image (like 'image/jpeg')
    file_extension: str, file extension of image, e.g. '.jpg'
    q_hash: str, hash of quantization matrices of image
    get_filetype_from_mime: bool, if True infer the file type from the
      mime. If False, infer filetype from file_extension.
    num_hash_digits: int, number of characters to retain in q_hash when
      form feature name.
  """
  
  filetype = get_filetype_name(mime, file_extension, get_filetype_from_mime)
  
  feature_name = filetype + "_" + str(compression_level).rjust(3,'0')   #e.g. "jpeg_090"
  
  if len(q_hash) > 0:
    feature_name += "_" + q_hash[0:num_hash_digits]       #e.g. "jpeg_090_6a1d3" 
    
  return feature_name

#==============================================================================

class attribution_quant_matrices:
  """Class to build a predictive model to verify an image's source.
  
  Implements a categorical Naive Bayes classifier, with features that include
  the mime type, image compression quality level, and (hashed) quantization
  matrices (for jpegs). Predictions are calibrated log likelihood ratios 
  (LLRs), but you can just treat them as scores for ROC curves.
  """
  def __init__(self):
    self.compr_categories = None
    self.num_compr_categories = None
    self.smooth_const = 1.0     #Laplace smoothing constant (for Naive Bayes model)
    self.known_sources = []
    self.model = {}
    self.full_train_size = 0
    self.initialized = False
  
  def fit(self, df_train, compr_category_names=None):
    global COMPR_NAME_UNKNOWN

    self.known_sources = sorted(list(df_train['source'].unique()))
    self.full_train_size = len(df_train['source'])

    if compr_category_names is None:
      #infer names of compression settings
      self.compr_categories = sorted(list(df_train['q_name'].unique()))
    else:
      #use provided list of compression settings. Also include categories detected in training data.
      self.compr_categories = sorted(list(set( list(compr_category_names) + list(df_train['q_name'].unique())  )))
    
    #Also include categories for error, and unrecognized compression setting
    self.compr_categories = sorted(list(set([COMPR_NAME_UNKNOWN] + self.compr_categories )))
    self.num_compr_categories = len(self.compr_categories)
    
    for source in self.known_sources:
      #Prepare data:
      #split train data into samples from the correct source, or from all other sources
      X_train_fromsource = df_train.loc[df_train['source']==source, 'q_name']
      X_train_notfromsource = df_train.loc[df_train['source']!=source, 'q_name']

      compression_counts_fromsource = X_train_fromsource.value_counts() #how many samples (from the source) have each compression setting?
      num_fromsource = compression_counts_fromsource.sum() #count total number of samples (from the source)

      compression_counts_notfromsource = X_train_notfromsource.value_counts() #how many samples (not from the source) have each compression setting?
      num_notfromsource = compression_counts_notfromsource.sum() #count total number of samples (not from the source)

      #---------------------------------------------------------------
      #From training data, compute model to predict the probability
      #of encountering a compression setting QS from a source
      prob_fromsource = {}
      for QS in self.compr_categories:
        count = self.smooth_const
        if QS in compression_counts_fromsource:
          count += compression_counts_fromsource[QS]
        prob_fromsource[QS] = count/(num_fromsource + self.smooth_const*self.num_compr_categories)

      prob_notfromsource = {}
      for QS in self.compr_categories:
        count = self.smooth_const
        if QS in compression_counts_notfromsource:
          count += compression_counts_notfromsource[QS]
        prob_notfromsource[QS] = count/(num_notfromsource + self.smooth_const*self.num_compr_categories)
      
      self.model[source] = {}
      self.model[source]['prob_fromsource'] = prob_fromsource
      self.model[source]['prob_notfromsource'] = prob_notfromsource
    
    self.initialized = True
      
  def _predict_from_source(self, compr_name, claimed_source):
    global COMPR_NAME_UNKNOWN
    
    if claimed_source not in self.known_sources:
      raise UnknownSourceError("{} is not a recognized source".format(claimed_source))
    
    if compr_name not in self.compr_categories:
      QN = COMPR_NAME_UNKNOWN 
      logging.debug("Unrecognized compression setting '{}'. Treating it as '{}'.".format(compr_name,QN))
    else:
      QN = compr_name

    prob_fromsource = self.model[claimed_source]['prob_fromsource'][QN]
    return prob_fromsource, (QN == COMPR_NAME_UNKNOWN)
      

  def _predict_not_from_source(self, compr_name, claimed_source):
    global COMPR_NAME_UNKNOWN

    if claimed_source not in self.known_sources:
      raise UnknownSourceError("{} is not a recognized source".format(self.claimed_source))
    
    if compr_name not in self.compr_categories:
      QN = COMPR_NAME_UNKNOWN 
      logging.debug("Unrecognized compression setting '{}'. Treating it as '{}'.".format(compr_name,QN))
    else:
      QN = compr_name

    prob_notfromsource = self.model[claimed_source]['prob_notfromsource'][QN]
    return prob_notfromsource, (QN == COMPR_NAME_UNKNOWN)
      
  def _predict_isfake_LLR(self, compr_name, claimed_source):
    if claimed_source not in self.known_sources:
      raise UnknownSourceError("{} is not a recognized source".format(self.claimed_source))

    prob_fromsource, unrecognized_compr1 = self._predict_from_source(compr_name, claimed_source)
    prob_notfromsource, unrecognized_compr2 = self._predict_not_from_source(compr_name, claimed_source)
    unrecognized_compr = unrecognized_compr1 or unrecognized_compr2

    LLR_isfake = np.log10(prob_notfromsource/prob_fromsource)
    return LLR_isfake, prob_fromsource, prob_notfromsource, unrecognized_compr

  def _predict_single(self, compr_name, claimed_source):
    return self._predict_isfake_LLR(compr_name, claimed_source)

  def _predict_multiple(self, compr_name, claimed_sources):
    if len(compr_name) != len(claimed_sources):
      raise ValueError("Args compr_name and claimed_sources have differing lengths")
    
    LLRs_isfake = []
    probs_fromsource = []
    probs_notfromsource = []
    unrecognized_sources = []
    for i in range(len(compr_name)):
      llr, pfs, pnfs, us = self._predict_single(compr_name[i], claimed_sources[i])
      LLRs_isfake.append(llr)
      probs_fromsource.append(pfs)
      probs_notfromsource.append(pnfs)
      unrecognized_sources.append(us)
    return LLRs_isfake, probs_fromsource, probs_notfromsource, unrecognized_sources

  def predict(self, compr_name, claimed_source):
    cn_arraylike = isinstance(compr_name,(list,pd.core.series.Series,np.ndarray))
    cs_arraylike = isinstance(claimed_source,(list,pd.core.series.Series,np.ndarray))
    if cn_arraylike != cs_arraylike:
      raise ValueError("Arguments must both be scalars or both array-like")
      
    if cn_arraylike:
      return self._predict_multiple(compr_name, claimed_source)
    else:
      return self._predict_single(compr_name, claimed_source)

  def predict_image(self, img_filepath, claimed_source):
    """Predicts if image is not from the claimed source.

    Args:
      img_filepath: str, path to image file to analyze.
      claimed_source: the claimed news source an image originated from.

    returns:
      float, LLR score where > 0 indicates misattribution, < 0 indicates
        correct attribution.

    Raises:
      ValueError: if the image is not a jpeg, according to the file mimetype.
      UnknownSourceError: if the source is not recognized
      FileNotFoundError: if the file does not exist
      
    """
    global COMPR_NAME_UNKNOWN

    if claimed_source not in self.known_sources:
      raise UnknownSourceError("{} is not a recognized source".format(self.claimed_source))

    compression_level, mime, file_extension, q_hash, _ = get_img_features(img_filepath)

    feature_name = make_feature_name(compression_level, mime, 
      file_extension, q_hash, get_filetype_from_mime=True)
      
    return self.predict(feature_name, claimed_source)

#==============================================================================
#New functions and classes used to support Experiment 4 in the paper

def encode_categorical_features(ds_train, extra_category_names = []):
  """Helper function to encode features before model fitting.
  
  Uses a label encoder to encode categorical features into numerical values,
  suitable for passing to sk-learn models.
  
  Args:
    ds_train: 1D pandas series (or list) of training data of category names
      to be encoded
    extra_category_names: list of additional categories, which may not be
      present in the training data, which you should nevertheless encode. 
  
  Returns:
    encoded_categories: categorical features after encoding
    le_feature: the sklearn preprocessing.LabelEncoder() that was fit
  """
  global COMPR_NAME_UNKNOWN
  
  feature_categories = sorted(list( ds_train.unique() ))
  #make sure we include the extra categories
  feature_categories = sorted(list(set(feature_categories +  extra_category_names)))

  le_feature = preprocessing.LabelEncoder()
  le_feature.fit(feature_categories)

  encoded_categories = le_feature.transform(ds_train)
  return encoded_categories, le_feature


class attribution_quant_matrices_sk:
  """WARNING: this class is different that prior classes.
    Prior classes sought to determine attribution or misattribution, which
    is binary classification, used in experiments 1-3.
    This class instead tries to predict the source class, which is N-class
    classification for N sources, used in experiment 4.  It uses the full 
    features including mime type, compression level, and quantization 
    matrices (for jpegs)."""

  def __init__(self):
    self.compr_categories = None
    self.num_compr_categories = None
    self.smooth_const = 1.0     #Laplace smoothing constant
    self.known_sources = []
    self.model = CategoricalNB()
    self.full_train_size = 0
    self.initialized = False
    
  def fit(self, X, y):
    """Fit Categorical Naive Bayes Model.
    
    Args:
      X: Matrix of features (encoded to categorical variables). Columns are 
        distinct features, rows are different data samples.
        WARNING: if using pandas, pass a dataframe (even if just 1 column) to
        avoid scikit-learn error, like this:
        df[['q_name_class']]       #not df['q_name_class']
      y: column of source labels, encoded as categorical variables.
    """

    self.full_train_size = len(y)
    
    self.model = CategoricalNB()
    self.model.fit(X, y)
    
    self.initialized = True
