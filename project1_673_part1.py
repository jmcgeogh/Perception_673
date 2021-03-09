#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:03:18 2021

@author: elliottmcg
"""
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage

  
vid = cv2.VideoCapture('Tag1.mp4') 
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count:', frame_count)
count = 0

# Function to scale the image
def scale(frame, percent):
    scale_percent = percent
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

# While loop that removes the background from each frame of the video
while True: 
    # Break while loop after checking last frame
    ret, frame = vid.read() 
    if ret:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
    # Scale the frame if needed
    resized = scale(frame, 100)
    
    if count == 100:
        # save org frame as JPEG file   
        cv2.imwrite("frame_org.jpg", resized)
        
    # Remove the background from the frame
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)
    
    for i in img_contours:
        if cv2.contourArea(i) > 100:
            break
        
    mask = np.zeros(resized.shape[:2], np.uint8)
    cv2.drawContours(mask, [i],-1, 255, -1)
    new_img = cv2.bitwise_and(resized, resized, mask=mask)
    
    # Show Video
    # cv2.imshow("Image with background removed", new_img) 
    # cv2.waitKey(1)
    
    # Select frame 99 from the video
    if count == 100:
        # save frame as JPEG file   
        cv2.imwrite("frame.jpg", new_img)     
    
    # Increase the count with the current frame number
    count += 1

cv2.destroyAllWindows()
###############
### Part 1a ###
# Grab saved frame
img = cv2.imread('frame.jpg',0)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Create magnitude spectrum with fft
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# Create mask
mask = np.ones((rows, cols, 2), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

fshift = dft_shift * mask
fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

# Inverse fft
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Plot the input image, magnitude spect, mask, and ifft
plt.subplot(2,3,1),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(mask_area, cmap = 'gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(fshift_mask_mag, cmap = 'gray')
plt.title('Magnitude Spectrum Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(img_back, cmap = 'gray')
plt.title('IFFT'), plt.xticks([]), plt.yticks([])
plt.show()

# Show selected frame and inverse fft
cv2.imshow("Selected frame", img)
cv2.waitKey(1)
cv2.imshow("ifft", img_back)
cv2.waitKey(1)
cv2.imwrite("ifft.jpg", img_back)
cv2.waitKey(0)

# Function to crop the image based on countours
def cropped(img, name):
    # Get image file
    img = cv2.imread(img)
    
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # threshold to get just the signature
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, \
                                        type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, \
                                               cv2.CHAIN_APPROX_SIMPLE)
            
    mx = (0,0,0,0)
    mx_area = 0
    # Create The bounding rectangle
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
            x,y,w,h = mx
    
    # Add buffers to the cropped image
    ey = int(h/3)
    ex = int(w/3)
    roi=img[y-ey:y+h+ey,x-ex:x+w+ex]
    roi = cv2.resize(roi,(1000,1000))
    
    # Show crop
    cv2.imshow(f'Cropped {name}', roi)
    cv2.imwrite(f"{name}_cropped.jpg", roi)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # Show bounding rectangle
    cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
    cv2.imshow(f'Detect Contours {name}', img)
    cv2.imwrite(f"{name}_boarder.jpg", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return

# Create crop of chosen frame and ifft
cropped('frame.jpg', 'frame')
cropped('ifft.jpg', 'ifft')
##############
## Part 1b ###
# Choose to used provided tage or frame from 1a
print("Choose Tag type to test")
print("press 1 for Refrence Marker")
print("press 2 for Chosen Frame")
a = int(input("\nMake your selection: "))
if a == 1:
    im = cv2.imread('ref_marker.png',0)
elif a == 2:
    im = cv2.imread('frame_cropped.jpg',0)
else:
    print("sorry selection could not be identified, exiting code")
    exit(0)

# Choose to flip the image
b = str(input("Do you wish to move the tag? (y/n) "))
if b == 'y':
    print("press 1 to flip vertically")
    print("press 2 to flip horizontally")
    print("press 3 to flip both horizontally and vertically")
    c = int(input("\nMake your selection: "))
    if c == 1:
        im = cv2.flip(im, 1)
    elif c == 2: 
        im = cv2.flip(im, 0)
    elif c == 3:
        im = cv2.flip(im, 1)
        im = cv2.flip(im, 0)
    else:
        print("sorry selection could not be identified, exiting code")
        exit(0)

# Resize the image
im = cv2.resize(im,(200,200))

# Average the pixels of the image to clean it up
m = np.array(im)
result = ndimage.generic_filter(m, np.nanmean, size=3, mode='constant', cval=np.NaN)
for x in range(len(result)):
    for y in range(len(result)):
        if result[x][y] >= 200:
            result[x][y] = 255
        else:
            result[x][y] = 0 

# resize the image
new_im = result
new_im = cv2.resize(new_im,(1000,1000))
cv2.imwrite('averaged_im.jpg', new_im)

# Get dim of the image
imgheight = new_im.shape[0]
imgwidth = new_im.shape[1]

# Break the image into an 8x8 grid
y1 = 0
M = imgheight//8
N = imgwidth//8

# Detect corners
corners = cv2.goodFeaturesToTrack(new_im,10,0.01,40)
corners = np.int0(corners)
corner_loc = []

# Create new image
tag = np.zeros((8,8))
tag = tag.astype(int)

# Display corner ditection and get coordinates
for i in corners:
    x,y = i.ravel()
    cv2.circle(new_im,(x,y),10,150,-1)
    corner_loc.append([x,y])

# Apply grid and lable the cells 0 and 1 fro black and white
for y in range(0,imgheight,M):
    for x in range(0, imgwidth, N):
        y1 = y + M
        x1 = x + N
        cv2.rectangle(new_im, (x, y), (x1, y1), (100, 0, 0))
        xc = int((x+x1)/2)
        yc = int((y+y1)/2)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (xc, yc) 
        fontScale = 1
        color = (100, 0, 0) 
        thickness = 2
        if new_im[yc,xc] >= 150:
            new_im = cv2.putText(new_im, '1', org, font,  
                              fontScale, color, thickness, cv2.LINE_AA)
            tag_x = int(x/N) 
            tag_y = int(y/M)
            tag[tag_y][tag_x] = 1
        else:
            new_im = cv2.putText(new_im, '0', org, font,  
                              fontScale, color, thickness, cv2.LINE_AA) 

# Get the tag code
code = ''
if tag[5][5] == 1:
        code = f'{tag[3][3]}{tag[3][4]}{tag[4][4]}{tag[4][3]}'
        loc = 'BR'
elif tag[5][2] == 1:
    code = f'{tag[3][4]}{tag[4][4]}{tag[4][3]}{tag[3][3]}'
    loc = 'BL'
elif tag[2][2] == 1:
    code = f'{tag[4][4]}{tag[4][3]}{tag[3][3]}{tag[3][4]}'
    loc = 'TL'
elif tag[2][5] == 1:
    code = f'{tag[4][3]}{tag[3][3]}{tag[3][4]}{tag[4][4]}'
    loc = 'TR'

# Print tag as a matrix and tag code in the console 
print('The current tag is: \n')
print(tag)
print('This tag code is: ' + code)
print(f'The orientation is {loc}')
cv2.imwrite("solution_1b.jpg", new_im)

# Show tag with the grid, numbers, and code
cv2.imshow(f"Tag code: {code}", new_im)
cv2.waitKey()

# Function to resize the tag matrix to show clean tag
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized
    dim = None
    (h, w) = image.shape[:2]
    
    # if both the width and height are None, then return the image
    if width is None and height is None:
        return image
    
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width
        r = width / float(w)
        dim = (width, int(h * r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    
    # return the resized image
    return resized

# Resize the tag matrix to show clean tag
tag_ar = np.array(tag[2:6, 2:6] * 255)
tag_img = tag_ar.astype(np.uint8)
tag_img = image_resize(tag_img, height = 1000)
cv2.imshow("Tag", tag_img)
cv2.imwrite("solution_1b_clean.jpg", tag_img)

cv2.waitKey()
cv2.destroyAllWindows()



 