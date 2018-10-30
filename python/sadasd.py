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

s = [u'+', u'+', u'o']
i = 0
markers = pd.DataFrame(predict)
markers = markers.replace(0, "+")
markers = markers.replace(1, "o")
