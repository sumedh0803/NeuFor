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

"""Generating Results"""
model = load('Recognize/recognize.joblib')
one = []
two = []
three = []
four = []
five = []

img_dir = "./Recognize/recognize_test/" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
from keras.preprocessing import image
for f1 in files:
    img = image.load_img(f1,target_size = (64,64))
    img = image.img_to_array(img)
    img = img/255
#len(img.shape) for getting the no of channels in the image
    img = np.expand_dims(img, axis = 0)
#result = classifier.predict_proba(img.flatten().reshape(-1,64,64,3))
    result = model.predict(img.flatten().reshape(-1,64,64,3))
    one.append(result[0][0])
    two.append(result[0][1])
    three.append(result[0][2])
    four.append(result[0][3])
    five.append(result[0][4])
    
    
addn = sum(one)
Suspect1 = addn/len(one)
Suspect1 = Suspect1 * 100
print(Suspect1)

addn = sum(two)
Suspect2 = addn/len(two)
Suspect2 = Suspect2 * 100
print(Suspect2)

addn = sum(three)
Suspect3 = addn/len(three)
Suspect3 = Suspect3 * 100
print(Suspect3)

addn = sum(four)
Suspect4 = addn/len(four)
Suspect4 = Suspect4 * 100
print(Suspect4)

addn = sum(five)
Suspect5 = addn/len(five)
Suspect5 = Suspect5 * 100
print(Suspect5)
#training_set.class_indices

Suspects = []
Suspects = pd.DataFrame(Suspects)
Suspects['Name'] = ['Suspect1','Suspect2','Suspect3','Suspect4','Suspect5']
Suspects['Probability'] = [Suspect1,Suspect2,Suspect3,Suspect4,Suspect5]
Suspects = Suspects.sort_values(by=['Probability'],ascending = False)
#Suspects.sort(reverse = 1)
#Delete the contents of the test folder
for root, dirs, files in os.walk('./Recognize/recognize_test/'):
    for f in files:
        os.unlink(os.path.join(root, f))

#Creating a dictionary of the top three suspects
data = {}  
data['Suspects'] = []  
data['Suspects'].append({  
    'Name': Suspects.iloc[0][0],
    'Probability': Suspects.iloc[0][1]
    
})
data['Suspects'].append({  
   'Name': Suspects.iloc[1][0],
   'Probability': Suspects.iloc[1][1]
})
data['Suspects'].append({  
    'Name': Suspects.iloc[2][0],
    'Probability': Suspects.iloc[2][1]
})

with open('data.txt', 'w') as outfile:  
    json.dump(data, outfile)

final = json.dumps(data)
print(int(Suspects.iloc[0][1]) - int(Suspects.iloc[1][1]))