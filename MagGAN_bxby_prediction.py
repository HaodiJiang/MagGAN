# =========================================================================
#   (c) Copyright 2022
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
from datetime import datetime, timedelta
from utils import *
create_predict_default_dirs()
import os
import sys

import tensorflow as tf
from MagGAN_Generator import danet_resnet101
# from model_loss import model_loss_
import numpy
import tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


result_dir = 'results_MagGAN'
model_dir = 'MagGAN_model'
data_dir = 'test_data'
sys.path.append(model_dir)
print('Models directory:', model_dir)
print('Data directory:', data_dir)
print('Will write result to:', result_dir)


tensorflow.config.experimental_run_functions_eagerly(True)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print('GPU Devices on the system:', gpu_devices)
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

starting_time = datetime.now().replace(microsecond=0)
print('Process start at:', starting_time)
modeX = danet_resnet101(256,256,1)
modeY = danet_resnet101(256,256,1)


print('Loading the model..')
bx_model = 'MagGAN_bx.h5'
by_model = 'MagGAN_by.h5'

modeX.load_weights(model_dir+'/'+bx_model)
modeY.load_weights(model_dir+'/'+by_model)

import cv2
print('Loading the data...')
if not os.path.exists(data_dir + os.sep + 'ha') or not os.path.exists(data_dir + os.sep + 'los'):
    print('Missing all or part of the data. Please check the ReadMe on how to download the data.')
    sys.exit()

files = os.listdir(data_dir + os.sep + 'ha//')
print('Number of files to work on:', len(files))
count = 1
print('Processing and predicting files, please wait..')
from astropy.io import fits
for files_h, files_l in zip(os.listdir(data_dir + os.sep + 'ha//'),os.listdir(data_dir + os.sep + 'los//')):
    if count % 10 == 0:
        print('Number of processed files:', count , 'of', len(files))
    ha_fits = fits.open(data_dir + os.sep + 'ha//' + files_h)
    ha_fits.verify('fix')
    ha_data = ha_fits[0].data
    ha_data = (ha_data - numpy.min(ha_data)) / (numpy.max(ha_data) - numpy.min(ha_data))
    ha = ha_data - numpy.mean(ha_data)

    los_fits = fits.open(data_dir + os.sep + 'los//' + files_l)
    los_fits.verify('fix')
    los_data = los_fits[0].data
    los = los_data / 200
    input_1 = numpy.expand_dims(ha,axis=0)
    input_1 = numpy.expand_dims(input_1,axis=3)
    input_2 = numpy.expand_dims(los,axis=2)
    input_2 = numpy.expand_dims(input_2,axis=0)
    input_ = [input_1,input_2]
    #output
    output_x = modeX.predict(input_)
    output_x = output_x[0,:,:,0]
    #output
    output_y = modeY.predict(input_)
    output_y = output_y[0,:,:,0]
    ##################
    output_img_x = output_x * 200
    output_img_y = output_y * 200
    output_img_x = numpy.flipud(output_img_x)
    output_img_y = numpy.flipud(output_img_y)
    # numpy.savetxt(result_dir+os.sep +'bx_'+'predict_'+files_l+'.csv', output_img_x, delimiter=',')
    # numpy.savetxt(result_dir+os.sep +'by_'+'predict_'+files_l+'.csv', output_img_y, delimiter=',')
    # plt.imsave(result_dir+os.sep +'bx_'+'predict_'+files_l+'.png',output_img_x, cmap='gray', vmin=-1000, vmax=1000)
    # plt.imsave(result_dir + os.sep +'by_'+'predict_'+files_l+'.png',output_img_y, cmap='gray', vmin=-1000, vmax=1000)
    # Save data to fits
    result_dir_bx = result_dir + '/bx'
    bx_fits_file = os.path.join(result_dir_bx, 'bx_{}'.format(files_l))
    try:
        os.remove(bx_fits_file)
    except OSError:
        pass
    bx_fits = fits.PrimaryHDU(output_img_x)
    bx_fits.writeto(bx_fits_file)

    result_dir_by = result_dir + '/by'
    by_fits_file = os.path.join(result_dir_by, 'by_{}'.format(files_l))
    try:
        os.remove(by_fits_file)
    except OSError:
        pass
    by_fits = fits.PrimaryHDU(output_img_y)
    by_fits.writeto(by_fits_file)

    count = count+1
ending_time = datetime.now().replace(microsecond=0)
diff_time = ending_time - starting_time
print('Total time to finish testing process:', (diff_time))
