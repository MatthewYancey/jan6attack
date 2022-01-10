# Jan 6 Attack Case Study


## Description
This repo contains the code demonstrating facial clustering on a selection of videos from the January 6 Capitol Attack. A walkthrough of the methodology can be found in this Medium article.

## Data Sources
All videos come from https://jan6attack.com/

## Code
**facial_embeddings.ipynb**: Loops through the videos and saves detected faces along with their embeddings to a dataframe.

**facial_clustering.ipynb**: Is used for clustering on found faces and exploring the results. 

**save_faces.ipynb**: Saves images of found faces.

**detection_and_clustering.py**: Several helper functions for face detection, clustering, and evaluation.

**general.py**: A set of helper functions for reading videos, transformaing dataframes, and other miscellaneous tasks.
