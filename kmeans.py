

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

N = 6

#np.random.normal(loc=5, scale=3, size=100)


def test_data(n):
    
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
    
    #df['label'] = np.random.choice(N, size=1000)
    return df


def calculate_centroids(df, n):
    centroids = [(0,0)]*n
    for i in range(n):
        centroids[i] = tuple(df[df.label==i].mean().loc[['x', 'y']])
    
    return centroids

def euclid_dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def closest_centroid(a, centroids):
    return np.nanargmin([euclid_dist(a, c) for c in centroids])

def relabel(df, n, centroids):
    active_centroids = [c for c in centroids if c==c]
    return df[['x', 'y']].apply(lambda c: closest_centroid(c, active_centroids), axis=1)


def kmeans(df, n, max_rounds=100, plot_results=False):
    df['label'] = np.random.choice(n, size=len(df))
    #assign initial random labels
    
    centroids = [(df.x.max() * np.random.random(), df.y.max() * np.random.random()) for i in range(N)]
    #initialize centroids randomly to avoid dying of centroids
    
    label_changes = 1
    #count the data points that have been reassigned. if those are 0, clusters converged.
    
    round_counter = max_rounds
    
    while label_changes!=0 and max_rounds!=0:
        check = df.label.copy(deep=True)
        centroids = calculate_centroids(df, n)
        df['label'] = relabel(df, n, centroids)
        label_changes = sum(check!=df.label)
        round_counter-=1
        if plot_results and (label_changes==0 or round_counter==0):
            print('Rounds for convergence: ', max_rounds-round_counter)
            print('Number of clusters left: ', len([c for c in centroids if c==c]))
            plt.scatter([i[0] for i in centroids], [i[1] for i in centroids], c='r')
            plt.scatter(df.x, df.y, c=df.label)
            plt.show()
    
    if round_counter==0:
        logging.warning('Round limit reached. Algorithm didnt converge.')

    if len([c for c in centroids if c==c]) < n:
        logging.warning('Nan-centroid detected. Number of clusters lower than n.')
    
    return df, centroids


df = test_data(30)

df, centroids = kmeans(df, N, plot_results=True)


