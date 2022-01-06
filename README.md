# Kitware Image Attribution


## Overview

This repo contains supplementary materials for the paper "Source Attribution of Online News Images by Compression Analysis", M. Albright, N. Menon, K. Roschke, and A. Basharat, IEEE WIFS 2021.  

**IEEE link**: [https://ieeexplore.ieee.org/document/9648385](https://ieeexplore.ieee.org/document/9648385)  
**DOI**: 10.1109/WIFS53200.2021.9648385

### About
In this work, we study how online news websites compress digital images in 
their articles.  We find that different sites often compress their images in
distinctive ways, which enable source attribution via analysis of image compression
settings. Further, we develop efficient classifiers that use image 
compression settings as features to perform news source verification and
identification, with high accuracy.  We also performed a survey of online news rooms and found that the majority of respondents report standardized photo publishing processes within their organization.

### Contents

The folder [dataset](dataset) contains the data used to perform the experiments in the paper, in the file called [data.csv](dataset/data.csv).  This file contains
compression settings, URLs, and other metadata from over 64k digital images scraped from over 34k online articles from 30 news sources.

The code used to generate experimental results is in the folder [image_compression_attribution/common/publications/2021-summer-attrib](image_compression_attribution/common/publications/2021-summer-attrib).  Note that the experiment notebooks make use of re-usable modules from the folder [image_compression_attribution/common/code](image_compression_attribution/common/code).

The folder [survey](survey) contains a detailed writeup (by Dr. Kristy Roschke) documenting the results of a photojournalism workflow survey.  The results from that survey were summarized in the published paper.


For further information please contact Dr. Arslan Basharat (arslan.basharat@kitware.com).

## Quickstart

We include a Dockerfile to conveniently run the experiment jupyter notebooks,
along with a Makefile to simplify the build and run steps.  We have configured
the Dockerfile to create a non-root user, and run as the non-root user, with
the user id matching the user who built the docker image.

To build the Docker image:
```shell
make docker-image
```

To run the jupyter notebook inside the docker container (with the jupyter 
notebook visible in your browser at `localhost:8888`), run:
```shell 
make run-jupyter
```

You can also run the docker container and access a bash shell with
```shell
make shell
```

## License

Permissive [BSD License](LICENSE).

## Citation
```
@inproceedings{albright2021attribution,
  title={Source Attribution of Online News Images by Compression Analysis},
  author={Michael Albright and Nitesh Menon and Kristy Roschke and Arslan Basharat},
  booktitle={2021 IEEE International Workshop on Information Forensics and Security (WIFS)},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgements: 
This research was developed with funding from the Defense Advanced Research Projects 
Agency (DARPA) under Contract No. HR001120C0123. The views, opinions and/or findings 
expressed are those of the author and should not be interpreted as representing the
official views or policies of the Department of Defense or the U.S. Government.  

Distribution Statement “A” (Approved for Public Release, Distribution Unlimited).
