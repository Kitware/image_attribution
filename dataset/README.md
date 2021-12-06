# Overview:

- data.csv is a copy of the processed dataset used in the experiments for the 
  WIFS 2021 publication. This summarizes compression settings from over 64k 
  images downloaded from over 34k online news articles from 30 news sources. 
  The first row of the csv lists the name of each column.

# Definition of column names:
The meanings of the columns in data.csv are as follows:
- articleUrl: url of the news article from which the image came
- articleHash: hash code to uniquely identify each article
- imageUrl: url of the news image
- source: name of the news source
- mime: file mime header (which indicates the file type, e.g. image/jpeg)
- compression: compression quality level of the image (integer from 1-100, estimated by ImageMagick)
- q\_hash: MD5 hash of the image's quantization matrice(s) - (for jpeg files only)
- q\_name: concatenation of the mime, compression, and q\_hash into one string summarizing all file and compression settings
- timestamp: date and time of the news article
