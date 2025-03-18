from flask import Flask, render_template, request, redirect, url_for
import turicreate as tc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from flask_frozen import Freezer


import json
import sys
import os
import random
import time

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import normalize

from copy import copy
from collections import defaultdict 
from itertools import combinations

app = Flask(__name__)

def sarray_to_json(x):
    return json.loads(x['tfidf'])


#KMEAN
def sframe_to_scipy_kmean(x, column_name):
    assert type(x[column_name][0]) == dict, \
        'Datatype not dict!'
    
    x = x.add_row_number('new_id')

    x = x.stack(column_name, ['feature', 'value'])

    unique_words = sorted(x['feature'].unique())
    mapping = {word:i for i, word in enumerate(unique_words)}
    x['feature_id'] = x['feature'].apply(lambda x: mapping[x])

    row_id = np.array(x['new_id'])
    col_id = np.array(x['feature_id'])
    data = np.array(x['value'])
    
    width = x['new_id'].max() + 1
    height = x['feature_id'].max() + 1
    
    mat = csr_matrix((data, (row_id, col_id)), shape=(width, height))
    return mat, mapping

def assign_clusters(data, centroids):
    distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)
    return cluster_assignment

def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        member_data_points = data[cluster_assignment==i]
        centroid = member_data_points.mean(axis=0)
        centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    
    return new_centroids

def compute_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in range(k):
        member_data_points = data[cluster_assignment==i, :]
        
        if member_data_points.shape[0] > 0: 
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
        
    return heterogeneity

def smart_initialize(data, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))

    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:].toarray()
    squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2
    
    for i in range(1, k):
        idx = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
        centroids[i] = data[idx,:].toarray()
       
        squared_distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean')**2,axis=1)
    
    return centroids

def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in range(maxiter):        
        if verbose:
            print(itr)

        cluster_assignment = assign_clusters(data, centroids)
        centroids = revise_centroids(data, k, cluster_assignment)        
        
        if prev_cluster_assignment is not None and \
          (prev_cluster_assignment==cluster_assignment).all():
            break
        
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]
        
    return centroids, cluster_assignment

def get_initial_centroids(data, k, seed=None):
    if seed is not None: 
        np.random.seed(seed)
    n = data.shape[0] # number of data points
        
    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)

    centroids = data[rand_indices,:].toarray()
    
    return centroids

def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    heterogeneity = {}
    
    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None
    
    for i in range(num_runs):
        if seed_list is not None: 
            seed = seed_list[i]
            np.random.seed(seed)
        else: 
            seed = int(time.time())
            np.random.seed(seed)
        
        initial_centroids = smart_initialize(data, k, seed=seed)
    
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter=400,
                                               record_heterogeneity=None, verbose=False)
        
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)
        
        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()
        
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment
     
    return final_centroids, final_cluster_assignment
        
def get_similar_docs_kmean(query_doc_id, centroids, cluster_assignment):
    id_to_index = {doc['id']: i for i, doc in enumerate(dataset1)}
    query_doc_index = id_to_index[query_doc_id]
    cluster_idx = cluster_assignment[k][query_doc_index]
    same_centroid_indices = [i for i, cluster in enumerate(cluster_assignment[k]) if cluster == cluster_idx]
    same_centroid_doc_ids = [list(id_to_index.keys())[list(id_to_index.values()).index(i)] for i in same_centroid_indices]
    similar_docs_sf = tc.SFrame({'id': same_centroid_doc_ids})
    similar_docs_sf = similar_docs_sf[similar_docs_sf['id'] != query_doc_id]
    similar_docs_sf = similar_docs_sf.head(10).join(dataset1[['id', 'topic', 'title', 'content']], on='id')
    return similar_docs_sf
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home_kmean')
def home_kmeans():
    return render_template('home_kmean.html', k_list=k_list)

app.route('/query_cluster_kmeans', methods=['POST'])
def query_cluster_kmeans():
    k = int(request.form['cluster'])
    centroid = centroids[k]
    distances = pairwise_distances(tf_idf, centroid, metric='euclidean')
    closest_docs = np.argsort(distances.flatten())[:10]
    closest_docs_list = closest_docs.tolist()
    closest_docs_sf = dataset1[dataset1['id'].apply(lambda x: x in closest_docs_list)]
    return render_template('cluster_docs.html', closest_docs=closest_docs_sf, k=k)

@app.route('/get_random_document_kmeans')
def get_random_document_kmeans():
    random_index = get_random_index()
    cluster_docs = get_similar_docs_kmean(random_index, centroids, cluster_assignment)
    return render_template('home_kmean.html', query_doc=dataset1[dataset1['id'] ==  random_index], cluster_docs=cluster_docs, k_list=k_list)

@app.route('/get_cluster_docs_kmeans/<int:k>')
def get_cluster_docs_kmeans(k):
    centroid = centroids[k]
    distances = pairwise_distances(tf_idf, centroid, metric='euclidean')
    closest_docs = np.argsort(distances.flatten())[:10]
    closest_docs_list = closest_docs.tolist()
    closest_docs_sf = dataset1[dataset1['id'].apply(lambda x: x in closest_docs_list)]
    return render_template('cluster_docs.html', closest_docs=closest_docs_sf, k=k)

def get_random_index():
    random_val = random.randint(0, 2000)
    return dataset1['id'][random_val]

#init
dataset1 = tc.SFrame.read_csv("dataset1.csv")
dataset1['tfidf'] = dataset1.apply(sarray_to_json)
random_index = get_random_index()

#init_Kmean
tf_idf, map_index_to_word = sframe_to_scipy_kmean(dataset1, 'tfidf')
tf_idf = normalize(tf_idf)
centroids = {}
cluster_assignment = {}
heterogeneity_values = []
#k_list = [2, 10, 25, 50, 100]
k_list = [3]
seed_list = [0]
#seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]
for k in k_list:
    heterogeneity = []
    centroids[k], cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, maxiter=400,
                                                               num_runs=len(seed_list), seed_list=seed_list,
                                                               verbose=True)
    score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
    heterogeneity_values.append(score)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)