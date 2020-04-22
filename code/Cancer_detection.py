from __future__ import print_function

import sys
sys.path.append('.\\')


import numpy as np
from keras import backend as K
import genome_handler
import devol
from keras.utils import HDF5Matrix

x_train = HDF5Matrix('D:\Xiankun\Research\\nor_set\\nor_train_x.h5', 'x',end = 10)
y_train = HDF5Matrix('D:\Xiankun\Research\pcamv1\\camelyonpatch_level_2_split_train_y.h5','y',end = 10)

x_validation = HDF5Matrix('D:\Xiankun\Research\\nor_set\\nor_valid_x.h5','x',end = 10)
y_validation = HDF5Matrix('D:\Xiankun\Research\pcamv1\\camelyonpatch_level_2_split_valid_y.h5','y',end = 10)

x_test = HDF5Matrix('D:\Xiankun\Research\\nor_set\\nor_test_x.h5','x',end = 10)
y_test = HDF5Matrix('D:\Xiankun\Research\pcamv1\\camelyonpatch_level_2_split_test_y.h5','y',end = 10)

y_train = np.squeeze(y_train)
y_validation = np.squeeze(y_validation)
y_test = np.squeeze(y_test)

dataset = ((x_train, y_train), (x_test, y_test),(x_validation, y_validation))



genome_handler = genome_handler.GenomeHandler(max_conv_layers=6,
                               max_dense_layers=3, # includes final dense layer
                               max_filters=512,
                               max_dense_nodes=512,
                               input_shape=x_train.shape[1:],
                               n_classes=1)



devol = devol.DEvol(genome_handler)
model = devol.run(dataset=dataset,
                  num_generations=1,
                  pop_size=1,
                  epochs=2)


print(model.summary())







