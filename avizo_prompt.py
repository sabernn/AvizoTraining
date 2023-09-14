import numpy as np
import os

y_pred_x5 = np.load('y_pred_x5_HDBSCAN.npy')
vol_x5 = y_pred_x5.reshape(80,46,46)

sf1 = hx_project.create('HxUniformScalarField3')

sf1.set_array(vol_x5)