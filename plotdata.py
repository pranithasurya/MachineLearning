# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:54:09 2015

@author: pranithasurya
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:26:30 2015

@author: pranithasurya
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

split=["90","50","10"]
datasets=["breast_cancer"]
path="datasets_v1/datasets_v1/"

colors=['k.','g.','r.','c.','m.','y.','w.','k.','g.']

#colors=['r.','k.']

for i in range(0,len(datasets),1):
    file=path+datasets[i]+"/data"
    data=np.loadtxt(file)
    data1=data
    file2=path+datasets[i]+"/trueclass"
    Y=np.loadtxt(file2,dtype=int)
    
    pca=PCA(n_components=2)
    X=pca.fit(data,Y).transform(data)    
    
    for k in range(0,len(X),1):
        plt.plot(X[k][0],X[k][1],colors[Y[k][0]],markersize=10)
    
    plt.show()
        
    kmeans=KMeans(n_clusters=2)
    kmeans.fit(data1)
    centroids=kmeans.cluster_centers_
    labels=kmeans.labels_        
    for k in range(0,len(data1),1):
        plt.plot(X[k][0],X[k][1],colors[labels[k]],markersize=10)
    #plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=150,linewidths=5,zorder=10)
        
    plt.show()
        
    kmeans=KMeans(n_clusters=8)
    kmeans.fit(data1)
    centroids=kmeans.cluster_centers_
    labels=kmeans.labels_        
    for k in range(0,len(data1),1):
        plt.plot(X[k][0],X[k][1],colors[labels[k]],markersize=10)
    #plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=150,linewidths=5,zorder=10)
        
    plt.show()
    
    db=DBSCAN(eps=50,min_samples=10).fit(data1)
    labels=db.labels_
    print(labels)
    for k in range(0,len(data1),1):
        plt.plot(X[k][0],X[k][1],colors[labels[k]],markersize=10)
    #plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=150,linewidths=5,zorder=10)
        
    plt.show()
 
    db=SpectralClustering(n_clusters=2,eigen_solver='lobpcg',affinity="nearest_neighbors")
    labels=db.fit_predict(data1)
    #print(labels)
    for k in range(0,len(data1),1):
        plt.plot(X[k][0],X[k][1],colors[labels[k]],markersize=10)
    #plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=150,linewidths=5,zorder=10)
        
    plt.show()   

   # row_num=np.arange(0,X.shape[0])
   # np.savetxt(path+datasets[i]+"/cluster_data",np.concatenate([labels.reshape(labels.size,-1),row_num.reshape(row_num.size,-1)],axis=1),fmt='%i')

    