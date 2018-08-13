import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

import time
from datetime import datetime

pd.set_option('display.max_columns', None)
# read csv
data = pd.read_csv("data/dataset.csv")
# data = data.drop(columns=['Unnamed: 0'])

data = data.drop(columns=['Unnamed: 0'])

print(data.head())
print(np.any(pd.isna(data)))
print(np.any(pd.isnull(data)))

data['timestamp'] = [time.mktime(
    pd.Timestamp.strptime(x, '%Y-%m-%d %H:%M:%S').timetuple())
    for x in data['block_timestamp']]
data = data.drop(columns=['block_timestamp'])

print(data.head())

model = KMeans(n_clusters=3, algorithm='auto')
model.fit(data)
predict = pd.DataFrame(model.predict(data))
predict.columns = ['predict']

r = pd.concat([data, predict], axis=1)
# plt.scatter(r['addrID'], r['txID'], c=r['predict'], alpha=0.5)



"""
centers = pd.DataFrame(model.cluster_centers_, columns=list(data))
center_x = centers['addrID']
center_y = centers['txID']
plt.scatter(center_x, center_y, s=50, marker='D', c='r')
plt.show()



# Plot ks vs inertias

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


"""

scaler = StandardScaler()
model = KMeans(n_clusters=7)
pipeline = make_pipeline(scaler, model)
pipeline.fit(data)
predict = pd.DataFrame(pipeline.predict(data))
predict.columns = ['predict']

# concatenate labels to df as a new column
r = pd.concat([data, predict], axis=1)

centers = pd.DataFrame(model.cluster_centers_, columns=list(data))
center_x = centers['addrID']
center_y = centers['txID']

plt.scatter(r['addrID'], r['txID'], c=r['predict'], alpha=0.7)
plt.scatter(center_x, center_y, s=50, marker='D', c='r')

plt.show()

ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(data)
    inertias.append(model.inertia_)

