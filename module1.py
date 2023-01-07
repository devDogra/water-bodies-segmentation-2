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

def show_image(image, title=None, cmap=None):
    plt.imshow(image, cmap=cmap)
    if title is not None: plt.title(title)
    plt.axis('off')

##################### MODEL CUSTOM OBJECTS ###############
# Contraction 
# @tf.keras.utils.register_keras_serializable()
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

# Expansion
# @tf.keras.utils.register_keras_serializable()
class DecoderBlock(keras.layers.Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock,self).__init__()
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)
    def call(self, inputs):
        X, short_X = inputs
        ct = self.up(X)
        c_ = concatenate([ct, short_X])
        x = self.net(c_)
        return x
 

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 
            "filters":self.filters,
            "rate":self.rate
        }

# Callback 
class ShowProgress(keras.callbacks.Callback):
    def __init__(self, SIZE):
        self.SIZE = SIZE
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(images))
        real_img = images[id][np.newaxis,...]
        pred_mask = self.model.predict(real_img)[0]
        mask = masks[id]

        plt.figure(figsize=(10,5))

        plt.subplot(1,3,1)
        show_image(real_img[0], title="Orginal Image")

        plt.subplot(1,3,2)
        show_image(pred_mask, title="Predicted Mask", cmap='gray')

        plt.subplot(1,3,3)
        show_image(mask, title="Orginal Mask", cmap='gray')


        plt.tight_layout()
        plt.show()

#########################################################

# Load model
new_model = tf.keras.models.load_model('UNet.h5', custom_objects={
    'EncoderBlock':EncoderBlock, 'DecoderBlock':DecoderBlock
})

#################################################333
SIZE = 128
# image_path = '.static/images/'
# mask_path = '.static/masks/'
# image_path = 'images/'
# mask_path = 'masks/'
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
    temp2_filename = img_filename.split('.')[0] +  '__UNET_MASK.jpg'
    # temp2_img.save('BOOOOOOOOGA.jpg')
    print(mask_path + temp2_filename)
    temp2_img.save(mask_path + temp2_filename)

    return mask_path + temp2_filename

#################################################

# just put the name of the image in the static folder, no need for path
# will save the mask in the static folder, with the name imgname_MASK.jpg
# predictAndSaveOutputFor('water_body_5.jpg')










# img_filename = 'water_body_10.jpg'

# path = image_path + img_filename
# img = img_to_array(load_img(path)).astype('float')/255
# img = cv.resize(img, (SIZE,SIZE), cv.INTER_AREA)
# # img_rs = img.reshape(1,SIZE,SIZE,3)
# imlist = [img]
# imlist = np.array(imlist)

# print(imlist.shape)
# print(img.shape)

# temp2 = new_model.predict(imlist)
# print(temp2.shape)

# flatten then reshape? 
# temp2 = temp2.flatten().reshape(SIZE,SIZE)

# print("AFER");
# print(temp2.shape); 

# print(temp2)


# from PIL import Image

# temp2 = temp2 * 255


# temp2 = temp2.astype(np.uint8)
# print(temp2)
# temp2_img = Image.fromarray(temp2)
# temp2_img.save('BOOOOOOOOGA.jpg')

# last resort wouldve been plt.savefig()
# after plt.imshow.
# fortunately, the PIL soln worked!








