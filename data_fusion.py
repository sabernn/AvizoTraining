

import os
import time
import warnings
import h5py

from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt

data_dir = "G:\\Saber\\WAAM\\Kyungmin_May5\\NP files"
data_dir = ""

Natten = np.load(os.path.join(data_dir, 'M1003_Natten.npy'))
NDFI = np.load(os.path.join(data_dir, 'M1003_NDFI.npy'))
XA = np.load(os.path.join(data_dir, 'M1003_X.npy'))

# with h5py.File(os.path.join(data_dir, 'M1003_XA.h5'), 'w') as f:
#     f.create_dataset('M1003_XA', data=XA)

XA = XA / (XA.max() - XA.min())

# print(Natten.dtype)
# print(NDFI.dtype)
# print(X.dtype)

# print(Natten.max(), Natten.min())
# print(NDFI.max(), NDFI.min())
# print(X.max(), X.min())

# print(Natten.sum())
# print(NDFI.sum())
# print(X.sum())


print(Natten.shape)
print(NDFI.shape)
print(XA.shape)

Natten_1D = Natten.reshape(-1)
NDFI_1D = NDFI.reshape(-1)
XA_1D = XA.reshape(-1)

AllIntensities = np.vstack((Natten_1D, NDFI_1D, XA_1D)).T

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# plt.xlabel('Neutron Attenuation')
# plt.ylabel('Neutron Dark Field Image')
# plt.clabel('X-ray Attenuation')
# plt.plot(Natten_1D[0::1000], NDFI_1D[0::1000], X_1D[0::1000],'.')
# plt.show()

# print(AllIntensities.shape)


# print(Natten_1D.shape)

slice_num = 200
max_slice_num = Natten.shape[0]

# fig = plt.figure()
# # plt.size = (30, 90)
# fig.add_subplot(1, 3, 1)
# plt.imshow(Natten[slice_num])
# plt.title(f'Neutron Attenuation, slice {slice_num}/{max_slice_num}')
# fig.add_subplot(1, 3, 2)
# plt.imshow(NDFI[slice_num])
# plt.title(f'Neutron Dark Field Image, slice {slice_num}/{max_slice_num}')
# fig.add_subplot(1, 3, 3)
# plt.imshow(X[slice_num])
# plt.title(f'X-ray Attenuation, slice {slice_num}/{max_slice_num}')

# plt.show()


# Clustering

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 2,
    "hdbscan_min_samples": 300,
}

params = default_base.copy()

dbscan = cluster.DBSCAN(eps=params["eps"])

hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        allow_single_cluster=params["allow_single_cluster"],
    )

t0 = time.time()

algorithm = hdbscan

X=np.stack((Natten_1D[0::1000], NDFI_1D[0::1000], XA_1D[0::1000]), axis=1)
# X=np.stack((Natten_1D[0::1000], NDFI_1D[0::1000], X_1D[0::1000]), axis=1)
# X=np.stack((Natten_1D, NDFI_1D, X_1D), axis=1)

# catch warnings related to kneighbors_graph
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="the number of connected components of the "
        + "connectivity matrix is [0-9]{1,2}"
        + " > 1. Completing it to avoid stopping the tree early.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding"
        + " may not work as expected.",
        category=UserWarning,
    )
    algorithm.fit(X)

t1 = time.time()
if hasattr(algorithm, "labels_"):
    y_pred = algorithm.labels_.astype(int)
else:
    y_pred = algorithm.predict(X)


colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )

print(y_pred.shape)
print(colors)
colorsplot = ['c','b','g','r']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], X[:,2], s=10, color=colors[y_pred])
ax.set_xlabel('Neutron Attenuation')
ax.set_ylabel('Neutron Dark Field Image')
# plt.zlabel('X-ray Attenuation')
ax.set_zlabel('X-ray Attenuation')

# plt.stem(y_pred)

plt.show()

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 2], s=10, color=colors[y_pred])
xlabel = 'Neutron Attenuation'
ylabel = 'X-ray Attenuation'
# plt.legend(loc='upper right')

# plt.xlim(-2.5, 2.5)
# plt.ylim(-2.5, 2.5)
# plt.xticks(())
# plt.yticks(())
plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )

plt.show()

cluster1 = X[y_pred==0]
cluster2 = X[y_pred==1]
cluster3 = X[y_pred==2]
cluster4 = X[y_pred==-1]

fig = plt.figure()
ax = fig.add_subplot(2,2,1,projection='3d')
ax.scatter3D(cluster1[:, 0], cluster1[:, 1], cluster1[:,2], s=10, color=colors[0])
ax.set_title('Cluster 1')
ax.set_xlabel('NA (avg = ' + str(np.mean(cluster1[:,0])) + ')')
ax.set_ylabel('NDFI (avg = ' + str(np.mean(cluster1[:,1])) + ')')
ax.set_zlabel('XA (avg = ' + str(np.mean(cluster1[:,2])) + ')')
ax.set_xlim(Natten_1D.min(), Natten_1D.max())
ax.set_ylim(NDFI_1D.min(), NDFI_1D.max())
ax.set_zlim(XA_1D.min(), XA_1D.max())

ax = fig.add_subplot(2,2,2,projection='3d')
ax.scatter3D(cluster2[:, 0], cluster2[:, 1], cluster2[:,2], s=10, color=colors[1])
ax.set_title('Cluster 2')
ax.set_xlabel('NA (avg = ' + str(np.mean(cluster2[:,0])) + ')')
ax.set_ylabel('NDFI (avg = ' + str(np.mean(cluster2[:,1])) + ')')
ax.set_zlabel('XA (avg = ' + str(np.mean(cluster2[:,2])) + ')')
ax.set_xlim(Natten_1D.min(), Natten_1D.max())
ax.set_ylim(NDFI_1D.min(), NDFI_1D.max())
ax.set_zlim(XA_1D.min(), XA_1D.max())

ax = fig.add_subplot(2,2,3,projection='3d')
ax.scatter3D(cluster3[:, 0], cluster3[:, 1], cluster3[:,2], s=10, color=colors[2])
ax.set_title('Cluster 3')
ax.set_xlabel('NA (avg = ' + str(np.mean(cluster3[:,0])) + ')')
ax.set_ylabel('NDFI (avg = ' + str(np.mean(cluster3[:,1])) + ')')
ax.set_zlabel('XA (avg = ' + str(np.mean(cluster3[:,2])) + ')')
ax.set_xlim(Natten_1D.min(), Natten_1D.max())
ax.set_ylim(NDFI_1D.min(), NDFI_1D.max())
ax.set_zlim(XA_1D.min(), XA_1D.max())

ax = fig.add_subplot(2,2,4,projection='3d')
ax.scatter3D(cluster4[:, 0], cluster4[:, 1], cluster4[:,2], s=10, color='red')
ax.set_title('Cluster 4 (outliers)')
ax.set_xlabel('NA (avg = ' + str(np.mean(cluster4[:,0])) + ')')
ax.set_ylabel('NDFI (avg = ' + str(np.mean(cluster4[:,1])) + ')')
ax.set_zlabel('XA (avg = ' + str(np.mean(cluster4[:,2])) + ')')
ax.set_xlim(Natten_1D.min(), Natten_1D.max())
ax.set_ylim(NDFI_1D.min(), NDFI_1D.max())
ax.set_zlim(XA_1D.min(), XA_1D.max())



plt.show()