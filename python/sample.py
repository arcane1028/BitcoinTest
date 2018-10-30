import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE

data_sample = pd.read_csv("../data/final_db_scale.csv", index_col=False)
# data_sample = data_in.sample(100000)
data_sample.head()
print(data_sample.shape)

model = KMeans(n_clusters=2, algorithm='auto')
model.fit(data_sample)
predict = pd.DataFrame(model.predict(data_sample))
predict.columns = ['predict']

result = pd.concat([data_sample, predict], axis=1)

centers = pd.DataFrame(model.cluster_centers_, columns=list(data_sample))
center_x = centers['addrID']
center_y = centers['sum']

# plt.yscale("log")
markers = pd.DataFrame(predict)
predict.columns = ['marker']

markers = markers.replace(0, "+")
markers = markers.replace(1, "o")

result = pd.concat([result, markers], axis=1)

i = 0

df1 = result[result['predict'] == 0]
df2 = result[result['predict'] == 1]

plt.scatter(df1['block_timestamp'], df1['balance'], c="b", alpha=0.5, marker="+")
plt.scatter(df2['block_timestamp'], df2['balance'], c="y", alpha=0.5, marker="o")

"""
for _m, c, _x, _y in zip(result["marker"], result['predict'], result['block_timestamp'], result['balance']):
    plt.scatter(_x, _y, marker=_m, c=c, alpha=0.5)
    print("{} th ".format(i))
    i += 1

"""
# plt.scatter(result['block_timestamp'], result['balance'], c=result['predict'], alpha=0.5, marker="+")
plt.scatter(center_x, center_y, s=50, marker='D', c='r')
plt.show()
