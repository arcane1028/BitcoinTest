import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle as pl

plt.switch_backend('agg')
pd.set_option('display.max_columns', None)

# read csv
print("read csv")
data_in = pd.read_csv("data_in.csv")
print(len(data_in.index))

data = data_in.sample(100000)
data = data.reset_index()

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
plt.savefig("./test.png")

with open('test.pkl', 'wb') as fid:
    pl.dump(fig_handle, fid)

# plt.show()

print("=====End=====")
