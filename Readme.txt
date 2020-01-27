#############################################################################################################################

Python version used: 3.7.0

Packages required to run:
i.   Pandas
ii.  Numpy
iii. Gensim
iv.  NLTK

Files Included:
i.   data_pre_processing.py        (Processes data and saves the vector representation of words in a CSV file)
ii.  k-means_clustering.py      (Performs the K-means clustering on the given dataset)
iii. DBSCAN.py                (An attempt to perform DBSCAN clustering)
iv.  doc2vec.csv              (Contains vector representation of words)

Running instructions (in Terminal):
** Run data_pre_processing.py file first. Generates a doc2vec.csv file used for K-Means clustering. It would take 30-40 minutes depending on your system.
** Then run k-means_clustering.py file. Generates a SSE score vs Number of cluster graph.
** Finally, run DBSCAN.py file as: python DBSCAN.py

#############################################################################################################################
