
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot  as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
# read csv
data = pd.read_csv("data2.csv")
data = data.drop(columns=['Unnamed: 0.1'])
print(data.head())

# np.any(np.isnan(data))
# np.all(np.isfinite(data))

#data = data.drop(columns=['block_timestamp'])
#data.dropna()

#test = data.sample(1000)
#print(test.head())

# check nan
print(np.any(np.isnan(data)))
print(np.all(np.isfinite(data)))

data = data.dropna()

print(np.any(np.isnan(data)))
print(np.all(np.isfinite(data)))

# create model and prediction
model = KMeans(n_clusters=7, algorithm='auto')
model.fit(data)
predict = pd.DataFrame(model.predict(data))
predict.columns = ['test']

r = pd.concat([data, predict], axis=1)
plt.scatter(r['txID'], r['sum'], c=r['test'], alpha=0.5)

centers = pd.DataFrame(model.cluster_centers_, columns=list(data))
center_x = centers['txID']
center_y = centers['sum']
plt.scatter(center_x, center_y, s=50, marker='D', c='r')
plt.show()

