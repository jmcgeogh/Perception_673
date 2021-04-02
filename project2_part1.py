#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:51:52 2021

@author: elliottmcg
"""

import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import math

# Call the video 
vid = cv2.VideoCapture('Night_Drive.mp4') 
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# Print number of frames in the video
print('Frame count:', frame_count)
done = 0

# Function to scale the image
def scale(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

# Create video frame generated frames
def visual(frames):
    path = ''
    img=[]
    for i in range(0,frames):
        img.append(cv2.imread(path+f'frame{i}.jpg'))
        
    height,width,layers=img[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name = 'Problem_1_Solution.avi'
    video=cv2.VideoWriter(name, fourcc, 15,(width,height))

    for j in range(0,len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()
    return

def gamma(frame):
    
    # Convert the image to HSV and split the channels
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    
    # Find gamma
    mid = 0.3
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)
    # print(gamma)
    
    # Do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)
    
    # Combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    return img_gamma

def cdf(gamma_frame):
    
    # Convert gamma frame to grayscale
    gray_gamma = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2GRAY)
    cols = gray_gamma.shape[1]
    rows = gray_gamma.shape[0]
    N = rows*cols
    # print(N)
    
    # Create list of values to use for histogram
    hist = []
    for x in range(cols):
        for y in range(rows):
            hist.append(gray_gamma[y][x])
    
    # Create dict for hist values
    freq = {}
    for item in hist:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    
    # Sort freq dict to put in correct order
    keylist = list(freq.keys())
    keylist.sort()
    # for key in keylist:
    #     print("%s : %s" % (key, freq[key]))
    
    # Create CDF curve values
    C = []
    for i in range(256):
        sum_ = 0
        for j in range(i+1):
            if j in freq:
                sum_ += freq[j]/N
            
        C.append(sum_)
    
    # Perform equalization using the CDF values and the hist list to create new img list
    new = []
    # print(C)
    for h in hist:
        new.append(C[h]*255)
    
    # Plot histogram, CDF curve, and equalized histogram
    fig, axs = plt.subplots(3)
    axs[0].hist(hist, bins = 255)
    axs[1].plot(range(256), C)
    axs[2].hist(new, bins = 255)
    plt.show()
    
    # Create the equalized image
    new_img = np.zeros([rows,cols,1],dtype=np.uint8)
    count = 0
    for x in range(cols):
        for y in range(rows):
            new_img[y][x] = new[count]
            count += 1
            
    return new_img

count = 0
# vid.set(1, 110)
# Test image for Gamma and Histogram Equailization
test_image = cv2.imread('test.jpg')

# Loop for Gamma Correction and Histogram Equalization
while True: 
    # Break while loop after checking last frame
    ret, frame = vid.read() 
    if ret:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
    # Resize Images
    resized = scale(frame, 25)
    # resized_test = scale(test_image, 25)
    
    # Call Gamma Correction Function
    img_gamma = gamma(resized)
    # test_gamma = gamma(resized_test)
    
    # Call Histogram Equization Function
    cdf_ = cdf(img_gamma)
    # cdf_test = cdf(test_gamma)
    
    # Show Outputs
    # resized_old = scale(frame, 100)
    # resized_gamma = scale(img_gamma, 400)
    # resized_test_gamma = scale(test_gamma, 400)
    # resized_cdf = scale(cdf_, 400)
    # resized_cdf_test = scale(cdf_test, 400)
    
    # cv2.imshow("Original", resized_old)
    # cv2.waitKey(1) 
    # cv2.imshow("Gamma", resized_gamma)
    # cv2.waitKey(1) 
    # cv2.imshow("CDF", resized_cdf)  
    # cv2.waitKey(1) 
    # cv2.imshow("Gamma Test", resized_test_gamma)
    # cv2.waitKey(1) 
    # cv2.imshow("CDF Test", resized_cdf_test)  
    # cv2.waitKey(1)  
    
    # Break loop after one frame
    # if done == 0:
    #     break
    # done += 1

    ###
    # Used in video creation, leave commented out
    # path = ''
    # cv2.imwrite((path+f'frame{count}.jpg'), img_gamma)
    # count += 1
    ###
    
###
# Used in video creation, leave commented out
# visual(count)
###

cv2.destroyAllWindows()