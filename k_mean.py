
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.switch_backend('agg')
pd.set_option('display.max_columns', None)

# read csv
print("read csv")
data = pd.read_csv("data_in.csv")
# data = data.drop(columns=['Unnamed: 0.1'])
print(len(data.index))
# np.any(np.isnan(data))
# np.all(np.isfinite(data))

# data = data.drop(columns=['block_timestamp'])
# data.dropna()

# test = data.sample(1000)
# print(test.head())

"""
# check nan
print(np.any(np.isnan(data)))
print(np.all(np.isfinite(data)))

data = data.dropna()

print(np.any(np.isnan(data)))
print(np.all(np.isfinite(data)))
"""

# create model and prediction
print("create k-mean")
scaler = StandardScaler()
model = KMeans(n_clusters=9, algorithm='auto')
pipeline = make_pipeline(scaler, model)
pipeline.fit(data)
predict = pd.DataFrame(pipeline.predict(data))
predict.columns = ['test']

print("create k-mean2")
r = pd.concat([data, predict], axis=1)
plt.scatter(r['addrID'], r['sum'], c=r['test'], alpha=0.5)

print("create k-mean3")
centers = pd.DataFrame(model.cluster_centers_, columns=list(data))
center_x = centers['addrID']
center_y = centers['sum']
plt.scatter(center_x, center_y, s=50, marker='D', c='r')
# plt.show()
plt.savefig("./test.png")
print("=====End=====")