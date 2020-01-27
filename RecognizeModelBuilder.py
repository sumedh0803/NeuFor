# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:47:48 2020

@author: sumed
"""


#Importing the libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from joblib import dump,load
from distutils.dir_util import copy_tree
import json

"""FOR SEGMENTING WORDS"""
fil_dir = os.getcwd()
seg_ctr = 0
char_counter = 0
img_dir = "./Recognize/suspect_images" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)


try:

    
    for f1 in files:
        sizeog = (os.stat(f1)).st_size
        #import image
        image = cv2.imread(f1)
        height, width, channels = image.shape
        #plt.imshow(image)
        #cv2.imshow('orig',image)
        #cv2.waitKey(0)
        
        #grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #plt.imshow(gray)
        #cv2.waitKey(0)
        
        #denoising image
        noisered = cv2.fastNlMeansDenoising(gray,None,10,7,21)
        #plt.imshow(dst)
        
        
        
        #binary
        """If pixel value is greater than a threshold value, 
        it is assigned one value (may be white), else it is assigned another value (may be black).
         The function used is cv.threshold. First argument is the source image, 
         which should be a grayscale image. Second argument is the threshold value 
         which is used to classify the pixel values. Third argument is the maxVal
         which represents the value to be given if pixel value is more than (sometimes less than)
         the threshold value."""
        """ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
        #cv2.imshow('second',thresh)
        #cv2.waitKey(100)"""
        
        th2 = cv2.adaptiveThreshold(noisered,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        #cv2.imshow('third',th2)
        #cv2.waitKey(0) 
        
        #dilation
        kernel = np.ones((2,7), np.uint8)
        img_dilation = cv2.dilate(th2, kernel, iterations=1)
        #cv2.imshow('dialated',img_dilation)
        #cv2.waitKey(0)
        
        #find contours
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
        
        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            if w>15 and h>15:
                # Getting ROI
                roi = image[y:y+h, x:x+w]
                #im = cv2.resize(roi,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
                #cv2.imshow('im',im)
                # show ROI
                cv2.imwrite('./Recognize/seg/'+str(seg_ctr)+'.jpg',roi)
                seg_ctr += 1
                #cv2.imshow('segment no:'+str(i),roi)
                #cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
                
         
               
        
        #cv2.imwrite('./output/marked areas26.jpg',image) #the output directory has to be manually made. else image wont be stored
        #cv2.waitKey(0)

        size = 0
        sizeog = 0.3 * sizeog
        
        
        
        for root, dirs, files in os.walk('./Recognize/seg'):
            for f in files:
                size = (os.stat('Recognize/seg/'+f)).st_size
                if(size >= sizeog):
                    os.unlink(os.path.join(root, f))
        
        """FOR SEGMENTING characters"""
        
        fil_dir = os.getcwd()
        
        
        for img_file in sorted(os.listdir(fil_dir + '/Recognize/seg/')):
            
            #import image
            #image = cv2.imread(fil_dir + '/seg/69.jpg',1)
            image = cv2.imread(fil_dir + '/Recognize/seg/' + img_file,1)
            #plt.imshow(image)
            #image = cv2.imread(fil_dir + '/seg/saurabh/129.jpg',1)
            height, width, channels = image.shape
            
            #denoising image
            dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
            
            #grayscale
            gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
            
            th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        #    cv2.imshow('third',th2)
        #    cv2.waitKey(0)
            
            #dilation
            kernel = np.ones((2,2), np.uint8)
            img_dilation = cv2.erode(th2, kernel, iterations=2)
            #plt.imshow(img_dilation)
            
            cols = []
            countp = []
            
            for i in range(width):
                temp = 0
                for j in range(height):
                    if img_dilation[j,i] == 255:
                        temp = temp + 1
                countp.append(temp)
                if temp <= 2:
                    cols.append(i)
                    
            for i in cols:
                image[:,i] = [255,0,0]
            
            #plt.imshow(image)
            
            #hist = plt.bar(range(width),countp)
            
            center = []
            start = 0
            
            for i  in range(len(cols)-1):
                end = i
                if abs(cols[i]-cols[i + 1]) > 10:
                    end = i
                    center.append(int((cols[start] + cols[end]) / 2))
                    start = i + 1
                
            center.append(int((cols[start] + cols[end]) / 2)) 
            #image = cv2.imread(fil_dir + '/seg/' + img_file,1)
            #image = cv2.imread(fil_dir + '/seg/69.jpg',1)
            for i in center:
                image[:,i] = [255,0,0]
            #cv2.imwrite('./seg/temp/char'+str(char_counter)+'.jpg',image)
            
            #image = cv2.imread(fil_dir + '/seg/' + img_file,1)    
            for i in range(len(center)):
                if i == len(center)-1:
                    cv2.imwrite('./Recognize/chars/char'+str(char_counter)+'.jpg',th2[:,center[i]:width])
                    char_counter += 1
                else:
                    cv2.imwrite('./Recognize/chars/char'+str(char_counter)+'.jpg',th2[:,center[i]:center[i+1]])
                    char_counter += 1
            
            char_counter += 1
    
except IndexError:
    pass
    
fromDirectory = "Recognize/chars"
toDirectory = "Recognize/recognize_test/"

copy_tree(fromDirectory, toDirectory)

for root, dirs, files in os.walk('./Recognize/seg/'):
    for f in files:
        os.unlink(os.path.join(root, f))

for root, dirs, files in os.walk('./Recognize/chars/'):
    for f in files:
        os.unlink(os.path.join(root, f))

for root, dirs, files in os.walk('./Recognize/suspect_images/'):
    for f in files:
        os.unlink(os.path.join(root, f))

#Convolutional Neural Network
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(5, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


#Data Augmentation
datagen = ImageDataGenerator(
        rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)  # randomly flip images

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         samples_per_epoch = 5286 ,
                         nb_epoch = 20,
                         
                         steps_per_epoch = 5286//32
                         )

dump(model, 'recognize.joblib')
