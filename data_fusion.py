

import os

import numpy as np
import matplotlib.pyplot as plt

data_dir = "G:\\Saber\\WAAM\\Kyungmin_May5\\NP files"
data_dir = ""

Natten = np.load(os.path.join(data_dir, 'M1003_Natten.npy'))
NDFI = np.load(os.path.join(data_dir, 'M1003_NDFI.npy'))
X = np.load(os.path.join(data_dir, 'M1003_X.npy'))
X = X / (X.max() - X.min())

print(Natten.dtype)
print(NDFI.dtype)
print(X.dtype)

print(Natten.max(), Natten.min())
print(NDFI.max(), NDFI.min())
print(X.max(), X.min())

print(Natten.sum())
print(NDFI.sum())
print(X.sum())


print(Natten.shape)
print(NDFI.shape)
print(X.shape)

Natten_1D = Natten.reshape(-1)
NDFI_1D = NDFI.reshape(-1)
X_1D = X.reshape(-1)

AllIntensities = np.vstack((Natten_1D, NDFI_1D, X_1D)).T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.xlabel('Neutron Attenuation')
plt.ylabel('Neutron Dark Field Image')
plt.clabel('X-ray Attenuation')
plt.plot(Natten_1D[0::1000], NDFI_1D[0::1000], X_1D[0::1000],'.')
plt.show()

print(AllIntensities.shape)


print(Natten_1D.shape)

slice_num = 200
max_slice_num = Natten.shape[0]

fig = plt.figure()
# plt.size = (30, 90)
fig.add_subplot(1, 3, 1)
plt.imshow(Natten[slice_num])
plt.title(f'Neutron Attenuation, slice {slice_num}/{max_slice_num}')
fig.add_subplot(1, 3, 2)
plt.imshow(NDFI[slice_num])
plt.title(f'Neutron Dark Field Image, slice {slice_num}/{max_slice_num}')
fig.add_subplot(1, 3, 3)
plt.imshow(X[slice_num])
plt.title(f'X-ray Attenuation, slice {slice_num}/{max_slice_num}')

plt.show()


