

import numpy as np
import os
import matplotlib.pyplot as plt
import h5py


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





# Create an HDF5 file
Atten_h5 = h5py.File('Atten.h5', 'w')
DFI_h5 = h5py.File('DFI.h5', 'w')
X_h5 = h5py.File('X.h5', 'w')


# Create a dataset in the HDF5 file
dset = Atten_h5.create_dataset('data', data=Atten)
dset = DFI_h5.create_dataset('data', data=DFI)
dset = X_h5.create_dataset('data', data=X)

# Close the HDF5 file
Atten_h5.close()
DFI_h5.close()
X_h5.close()




# import h5py
# import numpy as np

# # Load the .npy file
# arr = np.load('my_data.npy')

# # Create an HDF5 file
# file = h5py.File('my_data.h5', 'w')

# # Create a dataset in the HDF5 file
# dset = file.create_dataset('data', data=arr)

# # Close the HDF5 file
# file.close()