"""Summarize compression features in a dataset, including quantization matrices"""
import os
import errno
import sys
import argparse
import subprocess
import logging
import pickle
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm  #works in notebooks or scripts
from sklearn import preprocessing

#logging.getLogger().setLevel(logging.DEBUG)

from image_compression_attribution.common.code.models import quant_matrices
from image_compression_attribution.common.code.models import compr_levels

def summarize_compression_features(input_folder, output_file=None, 
  parse_meta_json=True, allow_missing_meta=False):
  """Summarize image compresssion settings in a collection of news images.
  
  The dataset should be a root-level folder. Inside that should be one
  folder for each news source, containing images from that source. This
  code generates a pickled dataframe summarizing compression settings.
  The summary includes compression levels and quantization matrices.
  
  Args:
    input_folder: str, path to folder containing input data.
      Inside input_folder, each folder should be a news source, e.g. 'bbc',
      containing images from that source.
    output_file: str or None, file path for a pickled dataframe summarizing the
      extracted compression settings in the dataset.  If None, do not save
      the pickled summary dataframe.
    parse_meta_json: bool, if True then parse the metadata json file (meta.json) 
      for each source to extract e.g. timestamp, URL, etc, and include that in 
      the generated data in the summary dataframe.
    allow_missing_meta: bool, if True then the code will allow meta.json to be
      missing for some sources.

  Returns:
    df: dataframe summarizing compression features from the dataset.
    error_messages: list of error messages.
    error_files: list of files that experienced errors.

  Raises:
    ValueError: if input_folder does not exist or is empty, or
      if parse_meta_json is True and the meta.json file does not exist.
    FileNotFoundError: if allow_missing_meta == False and meta.json is missing 
      inside any image folder.
  """
  if not os.path.isdir(input_folder):
    raise ValueError("{} is not a folder".format(input_folder))

  df = None
  error_messages = []
  error_files = []
  
  folder_names = sorted(next(os.walk(input_folder))[1])

  if len(folder_names) == 0:
    raise ValueError("No folders found in {}".format(input_folder))

  for folder_name in tqdm(folder_names):
    folder_path = os.path.join(input_folder, folder_name)
    file_names = sorted(next(os.walk(folder_path))[2])

    #Skip json metadata files. Only extract image features from image files.
    file_names = [x for x in file_names if not x.endswith(".json")]
    file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
    file_exts = [ os.path.splitext(file_name)[1] for file_name in file_names]
    compression_levels = []
    mimes = []
    file_longnames = []
    q_hashes = []
    feature_names = []
    
    for i, img_file in enumerate(tqdm(file_paths)): 
      try:
        compression_level, mime, file_extension, q_hash, _ = quant_matrices.get_img_features(img_file)

        feature_name = quant_matrices.make_feature_name(compression_level, mime, 
          file_extension, q_hash, get_filetype_from_mime=True)
        
      except Exception as e:
        compression_level = -1  #error
        q_hash = ""
        feature_name = quant_matrices.COMPR_NAME_ERROR      #error, e.g. "E"
        mime = compr_levels.get_mime(img_file) #need to recompute
        
        err_msg = "Error on file {}\n Error message: {}\n".format(img_file, str(e))
        error_files.append(img_file)
        error_messages.append(err_msg)
        logging.debug(err_msg)
      compression_levels.append(compression_level)
      mimes.append(mime)
      file_longnames.append( '{}/{}'.format(folder_name, file_names[i]) )
      q_hashes.append(q_hash)
      feature_names.append(feature_name)
      
    df_tmp = pd.DataFrame({"filename":file_names, "filepath":file_longnames,
      "compression":compression_levels, 'ext':file_exts, 'mime':mimes,
      "q_hash":q_hashes, "q_name":feature_names})
    df_tmp['source'] = folder_name

    #--------------------------------------------------------------------------
    #Optionally, load extra metadata about the dataset from meta.json files

    if parse_meta_json:
      meta_file = os.path.join(folder_path, "meta.json")

      if os.path.exists(meta_file): 

        df_meta = pd.read_json(meta_file)
        df_meta["filepath"] = folder_name + "/" + df_meta['filename']
        df_meta = df_meta[["filepath", "articleUrl", "articleHash", "imageUrl", "timestamp"]]
      
        df_tmp2 = pd.merge(left=df_tmp, right=df_meta, on='filepath', how='inner')
        #make sure the inner join doesn't kill any rows due to images with missing metadata
        assert len(df_tmp2) == len(df_tmp) 
        df_tmp = df_tmp2
      else: #no meta.json file
        if allow_missing_meta:
          #Fill in empty strings for the missing columns:
          df_tmp["articleUrl"] = ""
          df_tmp["articleHash"] = ""
          df_tmp["imageUrl"] = ""
          df_tmp["timestamp"] = ""
        else:
          raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), meta_file)
    #--------------------------------------------------------------------------

    if df is None:
      df = df_tmp
    else:
      df = pd.concat([df, df_tmp])

  #create numerical class labels for sources (suitable for use as training feature vector)
  sources = sorted(list(df['source'].unique()))
  le_sources = preprocessing.LabelEncoder()
  le_sources.fit(sources)
  df['source_class'] = le_sources.transform(df['source'])

  #Fix indices of dataframe which are repeating
  df.reset_index(drop=True, inplace=True)

  #Optionally save a copy of the summary dataframe to disk
  if output_file is not None:
    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)   
    df.to_pickle(output_file)

  return df, error_messages, error_files

def prune_df(df, max_num_images_per_source, summary_file_pruned=None):
  """Keep first N images from each source to create a pruned summary file.
  
    The motivation is to trim down samples from sources with too many images,
    to create more balanced datasets.

  Args:
    df: summary dataframe created by summarize_compression_features()
    max_num_images_per_source: int, keep this number of images (at max) from
      each source
    summary_file_pruned: if not None, save a copy of the pruned dataframe
      at this file path
  
  Returns:
    df_pruned: slimmer (pruned) dataset
  """
  df_pruned = df.groupby('source').head(max_num_images_per_source).reset_index(drop=True)
  if summary_file_pruned is not None:
    df_pruned.to_pickle(summary_file_pruned)  
  return df_pruned


if __name__ == "__main__":
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data/news_images")

  parser = argparse.ArgumentParser(description=
    'Summarize compression levels in dataset containing folder of images.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument("-i", "--in_folder", type=str, 
    default=os.path.join(DEFAULT_DATA_DIR, "raw"),
    help="Input folder to analyze, which contains folders of images from sources")

  parser.add_argument("-o", "--out_file", type=str, 
    default=os.path.join(DEFAULT_DATA_DIR, "processed/summary-quantization-features.pkl"),
    help="Path to output summary file to generate")

  parser.add_argument("-p", "--parse_meta", type=bool, default=True,
    help="If True, parse meta.json files to capture URLs and timestamps")

  #This is useful e.g. if you want to include GAN images in the training
  #(which have no article meta-data), in addition to images from real news 
  #articles (which do have article meta-data)
  parser.add_argument("-a", "--allow_missing_meta", type=bool, default=False,
    help="If True, allow missing meta.json files for some sources")

  parser.add_argument("-m", "--max_images", default=None,
    help="if not None, save a pruned summary keeping this many images per source.")

  parser.add_argument("-f", "--pruned_filename", 
    default=os.path.join(DEFAULT_DATA_DIR, "processed/summary-quantization-features-pruned.pkl"),
    help="if not None, save pruned summary dataframe to this file.")

  args = parser.parse_args()

  df = summarize_compression_features(args.in_folder, args.out_file, args.parse_meta, args.allow_missing_meta)

  if args.max_images is not None:
    assert type(args.max_images)==int
    #Keep first args.max_images images from each source and save a pruned summary file
    prune_df(df, args.max_images, args.pruned_filename)
    