# kitware-image-compression-attribution

## Contents
1. The folder [publications](publications) holds the code to perform the 
experiments for the publication. Those experiments are contained in Jupyter
notebooks. 
2. The folder [code](code) contains re-usable classes and functions used in
the publication notebooks to compute results.  Look here for details about
how compression features are extracted and how predictive models are built.

## Instructions to prepare data for training models

You can use the included code to summarize the image compression settings
from a collection of folders of images (from different news sources)
and save the summary file.  We used this functionality to create the 
dataset of summarized compression features released with this code.   


(Note that the following code saves the data summary as a pickle file, but we
 converted the official dataset to csv, and modified the experiment
 notebooks to read csv files, for easier sharing of the dataset.  If you
 process your own datasets, you will need to make minor changes e.g. to
 switch from pickle to csv.)

The input is a folder (./data/news_images_per_source/raw) which contains 
multiple folders; the name of each folder should be the name of the news source,
and the folder should contain images from that source.
The following command, run from the root level of this repository, can extract 
compression features and generate a dataset summary suitable for 
image attribution experiments:
```shell
python -m code.summarize_quant_matrices \
    -i ./data/news_images_per_source/raw \
    -o ./data/news_images_per_source/processed/summary-quantization-features.pkl \
    --parse_meta=False 
```



