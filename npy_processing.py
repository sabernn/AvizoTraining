

import numpy as np
import os
import matplotlib.pyplot as plt

DIR = "G:\\Saber\\WAAM\\Saber_Sep27\\NPY"

Atten = np.load(os.path.join(DIR, "Atten.npy"))
DFI = np.load(os.path.join(DIR, "DFI.npy"))
X = np.load(os.path.join(DIR, "X.npy"))

print(Atten.shape)


plt.subplot(1,3,1)
plt.imshow(Atten[:,:,300])
plt.subplot(1,3,2)
plt.imshow(DFI[:,:,300])
plt.subplot(1,3,3)
plt.imshow(X[:,:,300])
plt.show()


plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(Atten[:,:,90*i])
plt.show()
