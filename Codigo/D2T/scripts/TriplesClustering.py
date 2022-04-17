from math import ceil
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans 
import warnings
from sklearn.decomposition import PCA 
import numpy as np
import warnings

warnings.filterwarnings('ignore')



class TriplesClustering:

    def __init__(self, triples):

        self.triples = triples

        self.vectorized_triples = []
        self.clusters = []
        

    def setTriples(self, triples):
        self.triples = triples

    def getTriples(self):
        return self.triples

    def getVectorizedTriples(self):
        return self.vectorized_triples

    def genClusters(self):
        
        if(len(self.triples['input'])>7):
            vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.1, max_df=1.0)

            self.vectorized_triples = vectorizer.fit_transform(self.triples['input'])

            # initialize kmeans with 3 centroids
            n_clusters= ceil(len(self.triples['input'])/7)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)


            # fit the model
            kmeans.fit(self.vectorized_triples)
            # store cluster labels in a variable
            self.clusters = kmeans.labels_
            
            # initialize PCA with 2 components
            pca = PCA(n_components=2, random_state=42)
            # pass our X to the pca and store the reduced vectors into pca_vecs
            pca_vecs = pca.fit_transform(self.vectorized_triples.toarray())
            # save our two dimensions into x0 and x1
            x0 = pca_vecs[:, 0]
            x1 = pca_vecs[:, 1]

            # assign clusters and pca vectors to our dataframe 
            self.triples['cluster'] = []
            self.triples['x0'] = []
            self.triples['x1'] = []

            self.triples['cluster'] = self.clusters
            self.triples['x0'] = x0
            self.triples['x1'] = x1

        else:
            self.triples['cluster'] = np.zeros(shape = len(self.triples['input']), dtype = int)


