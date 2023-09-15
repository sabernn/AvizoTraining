

import numpy as np

vec_clust = np.load('y_pred_x5_HDBSCAN.npy')
# vol_clust = vec_clust.reshape(80,46,46)

Natten = np.load('M1003_Natten.npy')
NDFI = np.load('M1003_NDFI.npy')
XA = np.load('M1003_X.npy')

Natten_x5_1D = Natten[0::5, 0::5, 0::5].reshape(-1)
NDFI_x5_1D = NDFI[0::5, 0::5, 0::5].reshape(-1)
XA_x5_1D = XA[0::5, 0::5, 0::5].reshape(-1)


ox_ind = np.where(vec_clust == -1)
vc_ind = np.where(vec_clust == 0)
bg_ind = np.where(vec_clust == 1)
al_ind = np.where(vec_clust == 2)


Natten_ox = Natten_x5_1D[ox_ind]
NDFI_ox = NDFI_x5_1D[ox_ind]
XA_ox = XA_x5_1D[ox_ind]

Natten_vc = Natten_x5_1D[vc_ind]
NDFI_vc = NDFI_x5_1D[vc_ind]
XA_vc = XA_x5_1D[vc_ind]

Natten_bg = Natten_x5_1D[bg_ind]
NDFI_bg = NDFI_x5_1D[bg_ind]
XA_bg = XA_x5_1D[bg_ind]

Natten_al = Natten_x5_1D[al_ind]
NDFI_al = NDFI_x5_1D[al_ind]
XA_al = XA_x5_1D[al_ind]



