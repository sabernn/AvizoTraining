

import os

import numpy as np

data_dir = "G:\\Saber\\WAAM\\Kyungmin_May5\\NP files"

Natten = np.load(os.path.join(data_dir, 'M1003_Natten.npy'))
NDFI = np.load(os.path.join(data_dir, 'M1003_NDFI.npy'))
X = np.load(os.path.join(data_dir, 'M1003_X.npy'))

print(Natten.shape)
print(NDFI.shape)
print(X.shape)

