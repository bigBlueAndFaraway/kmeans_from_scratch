

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

N = 10



def create_test_data(n):
    '''gives a randomly generated cluster-like dataset
    input: n, number of clusters
    output: df, pd DataFrame with columns x and y. data is somewhat distributed
        in clusters of different mean/position and variance (normal distribution)
    '''
    
    #generate random means and variances.
    means_x = np.random.choice(range(10*n),size=n)
    means_y = np.random.choice(range(10*n),size=n)
    var_x = np.random.choice(range(n),size=n)
    var_y = np.random.choice(range(n),size=n)
    
    data_x = np.random.normal(loc=means_x[0], scale=var_x[0], size=100)
    data_y = np.random.normal(loc=means_y[0], scale=var_y[0], size=100)
    
    df = pd.DataFrame(data={'x': data_x, 'y': data_y})
    
    for i in range(1,n):
        data_x = np.random.normal(loc=means_x[i], scale=var_x[i], size=100)
        data_y = np.random.normal(loc=means_y[i], scale=var_y[i], size=100)
        df = pd.concat([df, pd.DataFrame(data={'x': data_x, 'y': data_y})], ignore_index=True)
    
    return df


def calculate_centroids(df, n):
    #caculate centroids as mean point of their respective cluster
    centroids = [(0,0)]*n
    for i in range(n):
        centroids[i] = tuple(df[df.label==i].mean().loc[['x', 'y']])
    
    return centroids

def euclid_dist(a, b):
    #returns euclidean distance of two 2d tuples
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def closest_centroid(a, centroids):
    #for a point a (2d tuple) returns the clostest point from centroids by euclid dist
    return np.nanargmin([euclid_dist(a, c) for c in centroids])

def relabel(df, n, centroids):
    #returns a pd series containing clostest centroid for each point in df
    #pd must have columns "x" and "y"
    active_centroids = [c for c in centroids if c==c]
    return df[['x', 'y']].apply(lambda c: closest_centroid(c, active_centroids), axis=1)


def kmeans(df, n, max_rounds=100, plot_results=False):
    '''core kmeans algorithm in its most basic form for 2d data.
    no efficiency improvements, no yinyang and the least elaborate initialization possible.
    
    In:
        df: a pandas dataframe with columns x and y
        n: number of clusters. output might have less than n clusters if some die off
        max_rounds: default 100, limit for calculation if no convergence achieved
        plot_results: default False, if True prints a scatterplot of clusters and cluster cores.

    Out:
        df: input pd dataframe with additional column "label"
        centroids: list of centroids
    '''
    
    #assign initial random labels
    df['label'] = np.random.choice(n, size=len(df))
    
    
    #initialize centroids randomly to avoid dying of centroids
    centroids = [(df.x.max() * np.random.random(), df.y.max() * np.random.random()) for i in range(N)]
    
    #count the data points that have been reassigned. if those are 0, clusters converged.
    label_changes = 1
    
    round_counter = max_rounds
    
    #the core loop of the alg, alternating between relabelling and recalculating centroids
    while label_changes!=0 and max_rounds!=0:
        
        #use "check" to check if convergence has been achieved by labels not changing
        check = df.label.copy(deep=True)
        
        centroids = calculate_centroids(df, n)
        df['label'] = relabel(df, n, centroids)

        label_changes = sum(check!=df.label)
        round_counter-=1

        if plot_results and (label_changes==0 or round_counter==0):
            print('Rounds for convergence: ', max_rounds-round_counter)
            print('Number of clusters left: ', len([c for c in centroids if c[0]==c[0] and c[1]==c[1]]))
            plt.scatter(df.x, df.y, c=df.label)
            plt.scatter([i[0] for i in centroids], [i[1] for i in centroids], c='r')
            plt.show()
    
    #if max rounds limit triggered stop, store info in log
    if round_counter==0:
        logging.info('Round limit reached. Algorithm didnt converge.')
        
    #if any centroids died off, store info in log
    if len([c for c in centroids if c[0]==c[0] and c[1]==c[1]]) < n:
        logging.info('Nan-centroid detected. Number of clusters lower than n.')
    
    return df, centroids



df = create_test_data(N)

df, centroids = kmeans(df, N, plot_results=True)

