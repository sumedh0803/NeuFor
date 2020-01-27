#Importing the libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os, os.path
from distutils.dir_util import copy_tree

"""FOR SEGMENTING WORDS""" 
fil_dir = os.getcwd()
seg_ctr = 0
char_counter = 0
img_dir = "./Match/known_samples" # Enter Directory of all images 
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
                cv2.imwrite('./Match/seg/'+str(i)+'.jpg',roi)
                #cv2.imshow('segment no:'+str(i),roi)
                #cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
                
         
               
        
        #cv2.imwrite('./output/marked areas26.jpg',image) #the output directory has to be manually made. else image wont be stored
        #cv2.waitKey(0)
        size = 0
        sizeog = 0.3 * sizeog
        
        
        
        for root, dirs, files in os.walk('./Match/seg'):
            for f in files:
                size = (os.stat('Match/seg/'+f)).st_size
                if(size >= sizeog):
            
                    os.unlink(os.path.join(root, f))
        
        """FOR SEGMENTING characters"""
        import os
        fil_dir = os.getcwd()
        char_counter = 0
        
        for img_file in sorted(os.listdir(fil_dir + '/Match/seg/')):
            
            #import image
            #image = cv2.imread(fil_dir + '/seg/69.jpg',1)
            image = cv2.imread(fil_dir + '/Match/seg/' + img_file,1)
            #plt.imshow(image)
            #image = cv2.imread(fil_dir + '/seg/saurabh/129.jpg',1)
            height, width, channels = image.shape
            
            #denoising image
            dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
            
            #grayscale
            gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
            
            th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
            #cv2.imshow('third',th2)
            #cv2.waitKey(0)
            
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
                    cv2.imwrite('./Match/chars/char'+str(char_counter)+'.jpg',th2[:,center[i]:width])
                    char_counter += 1
                else:
                    cv2.imwrite('./Match/chars/char'+str(char_counter)+'.jpg',th2[:,center[i]:center[i+1]])
                    char_counter += 1
            
            char_counter += 1

except IndexError:
    pass

"""Directory Adjustment"""

fromDirectory = "Match/chars"
toDirectory = "Match/training_set/known/"

copy_tree(fromDirectory, toDirectory)

for root, dirs, files in os.walk('./Match/seg/'):
    for f in files:
        os.unlink(os.path.join(root, f))

for root, dirs, files in os.walk('./Match/chars/'):
    for f in files:
        os.unlink(os.path.join(root, f))



class_fin = []
category = []
img_fin = []
flag = 1
flag2 = 0


"""onlyfiles = next(os.walk("./Match/training_set/suspect"))[2] #dir is your directory path as string
no_samples = len(onlyfiles)
if no_samples > 605:
    target = "suspect"
    targetval = no_samples
else:
    target = "zzzzz"
    targetval = 605


import os
diff = abs(605 - no_samples)
for root, dirs, files in os.walk('./Match/training_set/'+target+"/"):
    delete = 1
    del(delete)
files = pd.DataFrame(files)
todelete = files.iloc[len(files)-diff:,:]
todelete = todelete[0].tolist()
#root = './training_set/'+target+"/"
for f in todelete:
    os.unlink(os.path.join(root, f))
"""
for root, dirs, files in os.walk('./Match/known_samples'):
    for f in files:
        os.unlink(os.path.join(root, f))

print(1)