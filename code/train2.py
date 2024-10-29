#!/usr/bin/env python
#title           :train.py
#description     :to train the model
#author          :Fang Wang
#date            :2020/05/16
#usage           :python train.py --options
#python_version  :3.7.4 

from Network import Generator
from keras.models import Model
from keras.layers import Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
from numpy import save
from numpy import load
import tensorflow.keras.backend as K
import tensorflow as tf
import xarray as xr

## checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
##
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
###

def my_MSE_weighted(y_true, y_pred):
  weights= tf.clip_by_value(y_true, K.log(0.1+1), K.log(100.0+1))
  return K.mean(tf.multiply(weights, tf.abs(tf.subtract(y_pred, y_true))))

#np.random.seed(100)
image_shape_hr = (156,192,1)
image_shape_lr=(13, 16, 1)
downscale_factor = 12
print("reading input data")
# load low resolution data for training
merra2_train=load('/home/minghong/scratch/SRDRN/training/merra2_train_log.npy').reshape(121466, 13, 16, 1)

# load high resolution data for training
stage4_train=load('/home/minghong/scratch/SRDRN/training/stage4_train_log.npy').reshape(121466, 156, 192, 1)

#load low resolution data for validation
merra2_val=load('/home/minghong/scratch/SRDRN/validation/merra2_val_log.npy').reshape(25003, 13, 16, 1)

#load high resolution data for validation
stage4_val=load('/home/minghong/scratch/SRDRN/validation/stage4_val_log.npy').reshape(25003, 156, 192, 1)
print("finish reading input data")
#****************************************************************************************

def train(epochs, batch_size):
    
    x_train_lr=merra2_train
    y_train_hr=stage4_train
    
    x_val_lr=merra2_val
    y_val_hr=stage4_val   
    
 #   loss=MSE_LOSS(image_shape_hr)
    
    batch_count = int(x_train_lr.shape[0] / batch_size)
    
    generator = Generator(image_shape_lr).generator()
    generator.compile(loss='mae', optimizer = Adam(lr=0.0001, beta_1=0.9), metrics=['mae', 'mse'])
    loss_file = open('losses.txt' , 'w+')
    loss_file.close()
 ##checkpoint
    checkpoint = ModelCheckpoint('gen_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    callbacks_list = [checkpoint]
    for e in range(1, epochs+1):
        
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        
        for _ in tqdm(range(batch_count)):
            
            rand_nums = np.random.randint(0, x_train_lr.shape[0], size=batch_size)
            
            x_lr = x_train_lr[rand_nums]
            y_hr = y_train_hr[rand_nums]

            gen_loss=generator.train_on_batch(x_lr, y_hr)
        gen_loss = str(gen_loss)
        val_loss = generator.evaluate(x_val_lr, y_val_hr, verbose=0)
        val_loss = str(val_loss)
        loss_file = open('losses.txt' , 'a') 
        loss_file.write('epoch%d : generator_loss = %s; validation_loss = %s\n' 
                        %(e, gen_loss, val_loss))
        
        loss_file.close()
        if e <=20:
            if e  % 5== 0:
                generator.save('gen_model%d.h5' % e)
        else:
             if e  % 10 == 0:
                generator.save('gen_model%d.h5' % e)
        
train(150, 2)


