#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file(Generator)
#author          :Fang Wang
#date            :2022/2/11
#usage           :from Network import Generator
#python_version  :3.7.4 

# Modules
from tensorflow import keras
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
from keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers import AveragePooling2D
#import tensorflow_addons as tfa

# Residual block
def res_block_gen(model, kernal_size, filters, strides, initializer):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", kernel_initializer=initializer)(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", kernel_initializer=initializer)(model)
    model = BatchNormalization(momentum = 0.5)(model)
        
    model = add([gen, model])
    
    return model
    

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape

    def generator(self):
        init = RandomNormal(stddev=0.02)
        
        gen_input = Input(shape = self.noise_shape)
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=init)(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	    
        gen_model = model
        
        # Using 16 Residual Blocks
        for index in range(16):
	        model = res_block_gen(model, 3, 64, 1, init)
            
	    
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=init)(model)
        model = BatchNormalization(momentum = 0.5)(model)
        model = add([gen_model, model])
	    
	    # Using 3 UpSampling Blocks
        model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=init)(model)
        model = UpSampling2D(size = 2)(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
        
        model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=init)(model)
        model = UpSampling2D(size = 3)(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    
        model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=init)(model)
        model = UpSampling2D(size = 2)(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	    
        model = Conv2D(filters = 1, kernel_size = 9, strides = 1, padding = "same", kernel_initializer=init)(model)
        #model= tfa.image.median_filter2d(model, filter_shape=(37,37))
        #model= AveragePooling2D(pool_size=(37,37), strides=(1,1), padding='valid')(model)
        #model = Activation("sigmoid")(model)
	   
        generator_model = Model(inputs = gen_input, outputs = model)
        
        return generator_model

#model_gen=Generator((13, 16, 1)).generator()
#model_gen.summary()
#from tensorflow.keras.utils import plot_model
#plot_model(model_gen)