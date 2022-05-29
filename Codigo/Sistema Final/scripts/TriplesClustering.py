from math import ceil
from sklearn.feature_extraction.text import TfidfVectorizer
from k_means_constrained import KMeansConstrained
import warnings
from sklearn.decomposition import PCA 
import numpy as np
import warnings
import nltk
from nltk.corpus import stopwords 
# nltk.download('punkt')
# nltk.download('stopwords')
import seaborn as sns 
from collections import Counter
warnings.filterwarnings('ignore')
import re
import pandas as pd



class TriplesClustering:

    def __init__(self, triples):
        """
        This function takes in a dataframe of triples and creates a new column called 'cleaned' which is
        the same as the 'input' column. It then creates a list of vectorized triples, a list of
        clusters, and two empty strings
        
        :param triples: a dataframe with the following columns:
        """
        triples['cleaned'] = triples['input']
        self.triples = triples
        self.vectorized_triples = []
        self.clusters = np.zeros(shape=len(self.triples['input']))
        self.vectorizer = ""
        self.km = ""
        

    def setTriples(self, triples):
        """
        This function takes a list of triples and sets the triples attribute of the object to the list of
        triples
        
        :param triples: a list of triples, where each triple is a list of three strings
        """
        self.triples = triples

    def getTriples(self):
        """
        It returns the triples.
        :return: The triples are being returned.
        """
        return self.triples

    def getVectorizedTriples(self):
        """
        It takes a list of triples and returns a list of triples where each triple is a list of three
        lists of integers
        :return: The vectorized triples.
        """
        return self.vectorized_triples
    
    def getClusters(self):
        """
        It returns the clusters
        :return: The clusters
        """
        return self.clusters

    def cluster_mayor6(self): 
        """
        > If the number of clusters is greater than 7, return True. Otherwise, return False
        :return: True or False
        """
        conteo=Counter(self.clusters)

        for clave in conteo:  
            valor=conteo[clave]
            if valor >6:
                return True

        return False
        

    def genClusters(self):
        """
        It takes a dataframe of triples, and clusters them into groups of 6 or less. 
        """
        
        self.clusters = np.zeros(shape = len(self.triples['cleaned']), dtype = int)
        self.triples['cluster'] = self.clusters

        if(self.cluster_mayor6()):
            self.vectorizer = TfidfVectorizer(norm='l2', sublinear_tf=True,)

            self.vectorized_triples = self.vectorizer.fit_transform(self.triples['cleaned'])

            n_clusters = ceil(len(self.triples['cleaned'])/6)
            
            kmeans = KMeansConstrained(n_clusters = n_clusters, random_state = 8,size_min=3,size_max=6)
            self.km = kmeans

             # fit the model
            kmeans.fit(self.vectorized_triples.toarray())
            # store cluster labels in a variable
            self.clusters = kmeans.labels_
            
            
            # initialize PCA with 2 components
            pca  = PCA(n_components=3, random_state=40)
            # pass our X to the pca and store the reduced vectors into pca_vecs
            pca_vecs = pca.fit_transform(self.vectorized_triples.toarray())
            # save our two dimensions into x0 and x1
            x0 = pca_vecs[:, 0]
            x1 = pca_vecs[:, 1]
            x2 = pca_vecs[:, 2]
            
            self.triples['cluster'] = self.clusters
            self.triples['x0'] = x0
            self.triples['x1'] = x1
            self.triples['x2'] = x2

           

        
    def get_top_keywords(self, n_terms): 
        """
        For each cluster, find the n terms that have the highest tf-idf score.
        
        :param n_terms: the number of terms to return for each cluster
        """
        """This function returns the keywords for each centroid of the KMeans""" 

        X = self.getVectorizedTriples()
        clusters = self.getClusters()
        df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster 
        terms = self.vectorizer.get_feature_names_out() # access tf-idf terms 
        for i,r in df.iterrows(): 
            print('\nCluster {}'.format(i)) 
            print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
        
    