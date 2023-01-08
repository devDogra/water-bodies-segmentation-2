import os
import cv2 as cv
import pandas as pd
from tqdm import tqdm

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras import Sequential

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import *

from tensorflow import keras

#######################################
# Model custom layers
# Expansion
# NOdte: We had to add **kwarg in init to make this work, 
# for both modules
class DecoderBlock(keras.layers.Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock,self).__init__()
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.c1 = Conv2D(self.filters, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop = Dropout(self.rate)
        self.c2 = Conv2D(self.filters, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')

    def call(self, inputs):
        X, short_X = inputs
        ct = self.up(X)
        c_ = concatenate([ct, short_X])
        x = self.c1(c_)
        y = self.c2(x)
        return y

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 
            "filters":self.filters,
            "rate":self.rate,
        }


class AttentionChannel(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(AttentionChannel, self).__init__(**kwargs)
        self.filters = filters
        
        self.C1 = Conv2D(filters, kernel_size=1, strides=1, padding='same', activation=None)
        self.C2 = Conv2D(filters, kernel_size=1, strides=2, padding='same', activation=None)
#         self.relu = keras.activations.ReLU()
        tf.keras.layers.Activation('relu')

        self.add = keras.layers.Add()
        self.C3 = Conv2D(1,kernel_size=1, strides=1, padding='same', activation='sigmoid')
        self.up = keras.layers.UpSampling2D()
        self.mul = keras.layers.Multiply()
        self.BN = BatchNormalization()
                
    def call(self, X):
        org_x, skip_g = X
        g = self.C1(org_x)
        x = self.C2(skip_g)
        x = self.add([g,x])
        x = self.C3(x)
        x = self.up(x)
        x = self.mul([x,skip_g])
        x = self.BN(x)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "filters":self.filters
        })
        return base_config


class EncoderBlock(keras.layers.Layer):

    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock,self).__init__()
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.conv1 = Conv2D(self.filters,kernel_size=3,strides=1,padding='same',activation='relu',kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters,kernel_size=3,strides=1,padding='same',activation='relu',kernel_initializer='he_normal')
        self.pool = MaxPool2D(pool_size=(2,2))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.pooling: 
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 
            "filters":self.filters,
            "rate":self.rate,
            "pooling":self.pooling
        }


################################################
new_model = tf.keras.models.load_model('AttentionUNet.h5', custom_objects={
    'EncoderBlock':EncoderBlock, 'DecoderBlock':DecoderBlock, 'AttentionChannel':AttentionChannel
})

SIZE = 128
image_path = './static/'
mask_path = './static/'


def predictAndSaveOutputFor(img_filename):
    path = image_path + img_filename
    img = img_to_array(load_img(path)).astype('float')/255
    img = cv.resize(img, (SIZE,SIZE), cv.INTER_AREA)
    # img_rs = img.reshape(1,SIZE,SIZE,3)
    imlist = [img]
    imlist = np.array(imlist)
    
    temp2 = new_model.predict(imlist)
    
    # Flatten then reshape
    temp2 = temp2.flatten().reshape(SIZE,SIZE)
    
    # New color map
    # cm = plt.get_cmap('gist_rainbow')
    # temp2 = cm(temp2)
    
    # Save output mask 
    from PIL import Image
    
    temp2 = temp2 * 255 
    temp2 = temp2.astype(np.uint8)  # Float -> Int
    # print(temp2)
    temp2_img = Image.fromarray(temp2)
    temp2_filename = img_filename.split('.')[0] +  '__ATTNUNET_MASK.jpg'
    # temp2_img.save('BOOOOOOOOGA.jpg')
    print(mask_path + temp2_filename)
    temp2_img.save(mask_path + temp2_filename)

    return mask_path + temp2_filename


################################################3
# predictAndSaveOutputFor('water_body_4.jpg')