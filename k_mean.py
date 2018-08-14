import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import sklearn as sl
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import pickle as pl

# plt.switch_backend('agg')
pd.set_option('display.max_columns', None)

# read csv
print("read csv")
data_in = pd.read_csv("data_in.csv", index_col=False)
print(len(data_in.index))

data_sample = data_in.sample(100000)
# data = data.reset_index()
data_sample.to_csv("data_in_sample.csv", encoding='utf-8', index=False)

scaled_data = StandardScaler().fit_transform(data_sample.values)
data = pd.DataFrame(scaled_data, columns=data_sample.columns)


# create model and prediction
print("create k-mean")
"""
scaler = StandardScaler()
model = KMeans(n_clusters=3, algorithm='auto')
pipeline = make_pipeline(scaler, model)
pipeline.fit(data)
predict = pd.DataFrame(pipeline.predict(data))
predict.columns = ['predict']

"""
ks = range(1, 20)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(data)
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

"""
model = KMeans(n_clusters=3, algorithm='auto')
model.fit(data)
predict = pd.DataFrame(model.predict(data))
predict.columns = ['predict']

print("concat")
r = pd.concat([data, predict], axis=1)

print("plot")
centers = pd.DataFrame(model.cluster_centers_, columns=list(data))
center_x = centers['addrID']
center_y = centers['sum']

fig_handle = plt.figure()
plt.scatter(r['addrID'], r['sum'], c=r['predict'], alpha=0.5)
plt.scatter(center_x, center_y, s=50, marker='D', c='r')
# plt.savefig("./test.png")
# with open('test.pkl', 'wb') as fid:
#    pl.dump(fig_handle, fid)
plt.show()
"""

print("=====End=====")
"""
# PCA
# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(data)
fig_handle = plt.figure()

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

model = PCA(n_components=9)
pca_features = model.fit_transform(data)
fig_handle = plt.figure()

xf = pca_features[:,7]
yf = pca_features[:,8]
plt.scatter(xf,yf,c=r['predict'])
plt.show()

# TSNE long
model = TSNE(learning_rate=100)
transformed = model.fit_transform(data)
xs = transformed[:,0]
ys = transformed[:,1]
fig = plt.figure()
plt.scatter(xs,ys,c=r['predict'])
plt.show()

"""