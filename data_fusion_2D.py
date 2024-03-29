

import os
import time
import warnings
import h5py

from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt
import cv2

plot = True

data_dir = "G:\\Saber\\WAAM\\Kyungmin_May5\\NP files"
data_dir = "MELD"


Natten = cv2.imread(os.path.join(data_dir, 'TI_ACL_0880nm_annot_8bit.tif'))
NDFI = cv2.imread(os.path.join(data_dir, 'DFI_ACL_0880nm_annot_8bit.tif'))
NDPC = cv2.imread(os.path.join(data_dir, 'DPC_ACL_0880nm_annot_8bit.tif'))

if plot:
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(Natten)
    fig.add_subplot(1, 3, 2)
    plt.imshow(NDFI)
    fig.add_subplot(1, 3, 3)
    plt.imshow(NDPC)
    
    plt.show()


# Natten = cv2.load(os.path.join(data_dir, 'TI_ACL_880nm.tiff'))
# NDFI = cv2.load(os.path.join(data_dir, 'DFI_ACL_880nm.tiff'))
# NDPC = cv2.load(os.path.join(data_dir, 'DPC_ACL_880nm.tiff'))

# with h5py.File(os.path.join(data_dir, 'M1003_XA.h5'), 'w') as f:
#     f.create_dataset('M1003_XA', data=XA)

# XA = XA / (XA.max() - XA.min())

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
print(NDPC.shape)

sf = 20  # scale factor

Natten_ds = Natten[0::sf, 0::sf, 0]
NDFI_ds = NDFI[0::sf, 0::sf, 0]
NDPC_ds = NDPC[0::sf, 0::sf, 0]



print(Natten_ds.shape)
print(NDFI_ds.shape)
print(NDPC_ds.shape)
print(Natten.shape)
print(NDFI.shape)
print(NDPC.shape)

Natten_1D = Natten_ds.reshape(-1)
NDFI_1D = NDFI_ds.reshape(-1)
NDPC_1D = NDPC_ds.reshape(-1)

# Natten_1D = Natten.reshape(-1)
# NDFI_1D = NDFI.reshape(-1)
# XA_1D = XA.reshape(-1)


AllIntensities = np.vstack((Natten_1D, NDFI_1D, NDPC_1D)).T


# if plot:
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     plt.xlabel('Neutron Attenuation')
#     plt.ylabel('Neutron Dark Field Image')
#     plt.clabel('X-ray Attenuation')
#     plt.plot(Natten_1D[0::1000], NDFI_1D[0::1000], XA_1D[0::1000],'.')
#     plt.show()

# print(AllIntensities.shape)


# print(Natten_1D.shape)

slice_num = 200
max_slice_num = Natten.shape[0]

if plot:
    fig = plt.figure()
    # plt.size = (30, 90)
    fig.add_subplot(1, 3, 1)
    plt.imshow(Natten[slice_num])
    plt.title(f'Neutron Attenuation, slice {slice_num}/{max_slice_num}')
    fig.add_subplot(1, 3, 2)
    plt.imshow(NDFI[slice_num])
    plt.title(f'Neutron Dark Field Image, slice {slice_num}/{max_slice_num}')
    fig.add_subplot(1, 3, 3)
    plt.imshow(NDPC[slice_num])
    plt.title(f'X-ray Attenuation, slice {slice_num}/{max_slice_num}')

    plt.show()


# Clustering

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

default_base = {
    "quantile": 0.3,
    "eps": 0.008,
    "damping": 0.3,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": False,
    "hdbscan_min_cluster_size": 1000,
    "hdbscan_min_samples": 300,
}

params = default_base.copy()

dbscan = cluster.DBSCAN(eps=params["eps"])

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        allow_single_cluster=params["allow_single_cluster"],
    )

t0 = time.time()


algorithm = hdbscan
# algorithm = dbscan

X=np.stack((Natten_1D, NDFI_1D, XA_1D), axis=1)
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
print(f"Clustering time: {t1 - t0:.2f}s")
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
np.save('y_pred_x5_HDBSCAN.npy', y_pred)
print(y_pred.min(), y_pred.max())
print(colors)
colorsplot = ['c','b','g','r']

if plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:,2], s=10, color=colors[y_pred])
    ax.set_xlabel('Neutron Attenuation')
    ax.set_ylabel('Neutron Dark Field Image')
#     plt.zlabel('X-ray Attenuation')
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


if plot:
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