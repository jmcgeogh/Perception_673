#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:36:43 2021

@author: elliottmcg
"""

import numpy as np
import cv2
import copy
from scipy import ndimage
import math


# Select video to test with
print("Choose from the selected options for Tag videos")
print("press 1 for Tag0")
print("press 2 for Tag1")
print("press 3 for Tag2")
print("press 4 for Multiple_tags")
print("")
a = int(input("Make your selection: "))
if a == 1:
    cap = cv2.VideoCapture('Tag0.mp4')
elif a == 2:
    cap = cv2.VideoCapture('Tag1.mp4')
elif a == 3:
    cap = cv2.VideoCapture('Tag2.mp4')
elif a == 4:
    cap = cv2.VideoCapture('multipleTags.mp4')
else:
    print("sorry selection could not be identified, exiting code")
    exit(0)

# Get frames of selected video
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count:', frame_count)

# Get Testudo image and resize
testu_img = cv2.imread('testudo.png')
testu_resize = cv2.resize(testu_img, (200, 200))

# Dimension out the warp of the tag
dim = 200
p1 = np.array([
    [0, 0],
    [dim - 1, 0],
    [dim - 1, dim - 1],
    [0, dim - 1]], dtype="float32")

# Initialize the location history and code history lists
loc_his1 = ['']
loc_his2 = ['']
loc_his3 = ['']
code_his1 = ['']
code_his2 = ['']
code_his3 = ['']

# Warp the tag
def warped(im, h, height, width, corners):
    # Create warp image
    warp = np.zeros((height, width,3), np.uint8)
    
    # Ensure singular matrices are skipped
    if np.linalg.det(h) != 0:
        # Invert Homogrphy Matrix
        h = np.linalg.inv(h)
        
        # Create warpped image by applying homography to frame
        for a in range(height):
            for b in range(width):
                old = [a,b,1]
                old = np.reshape(old, (3,1))
                x,y,z = np.matmul(h, old)
                # Convert values to float to be usable
                xi = float(x)
                yi = float(y)
                zi = float(z)
                img_x = 0
                img_y = 0
                
                # Ensure no div by zero
                if zi != 0:
                    # Ensure no float infinity
                    if math.isinf((xi/zi)) == False and math.isinf((yi/zi)) == False:
                        img_x = abs(int(x/z))
                        img_y = abs(int(y/z))
                # Ensure mapped points are viable                   
                if img_y < im.shape[0] and img_x < im.shape[1]:
                    warp[b][a] = im[img_y][img_x]
            
    return warp

# Warp the testudo
def testu_warped(img, h, frame, corners):
    height = frame.shape[0]
    width = frame.shape[1]
    warp = np.zeros((height, width,3), np.uint8)
    if np.linalg.det(h) != 0:
        h = np.linalg.inv(h)
        
        # Find and sort corner points 
        ys = []
        xs = []
        for i in range(len(corners)):
            ys.append(corners[i][0])
            xs.append(corners[i][1])
            
        ys.sort()
        xs.sort()
        # Only use corner points to reduce run time and stop multiple images
        for a in range(int(xs[0]), int(xs[-1])):
            for b in range(int(ys[0]), int(ys[-1])):
                old = [b,a,1]
                old = np.reshape(old, (3,1))
                x,y,z = np.matmul(h, old)
                xi = float(x)
                yi = float(y)
                zi = float(z)
                if not math.isinf((xi/zi)) or math.isinf((yi/zi)):
                    img_x = abs(int(x/z))
                    img_y = abs(int(y/z))
                else:
                    img_x = 0
                    img_y = 0
                    
                if img_y < img.shape[0] and img_x < img.shape[1]:
                    warp[a][b] = img[img_y][img_x]
            

    return warp

# Get the code of the tag and the orientation
def code(im, his_num):
    # Resize image to reduce run time
    im = cv2.resize(im,(64,64))
    m = np.array(im)
    
    # Average pixel filter to clean up image
    result = ndimage.generic_filter(m, np.nanmean, size=3, mode='constant', cval=np.NaN)
    for x in range(len(result)):
        for y in range(len(result)):
            if result[x][y] >= 200:
                result[x][y] = 255
            else:
                result[x][y] = 0 
                
    new_im = result
    
    # Break image into 8x8 grid to detect code and orientation
    imgheight = new_im.shape[0]
    imgwidth = new_im.shape[1]
    M = imgheight//8
    N = imgwidth//8
    
    # Create tag mapping matrix
    tag = np.zeros((8,8))
    tag = tag.astype(int)

    # Search average image and create tage matrix
    # where black tile are 0 and whie tiles are 1
    for y in range(0,imgheight,M):
        for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N
            xc = int((x+x1)/2)
            yc = int((y+y1)/2)
            if new_im[yc,xc] >= 150:
                tag_x = int(x/N) 
                tag_y = int(y/M)
                tag[tag_y][tag_x] = 1 
    
    # Detect which history list to call       
    if his_num == 1:
        old_loc = loc_his1
        old_code = code_his1
    if his_num == 2:
        old_loc = loc_his2
        old_code = code_his2
    if his_num == 3:
        old_loc = loc_his3
        old_code = code_his3
    
    code = ''
    loc = ''
    # Create a of the inner 4x4 corner values from the tag matrix 
    count = 0
    corners = [tag[5][5], tag[5][2], tag[2][2], tag[2][5]]
    
    # If there is only one corner value equalling 1 then it is a clean read
    # If not default to the most recent clean read
    for i in corners:
        if i == 1:
            count += 1
    if 0 < count <= 1:
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
    else:
        if old_code[-1] == '':
            code = None
            loc = None
        else:
            code = old_code[-1]
            loc = old_loc[-1]
    
    return code, loc


# Order of points from the contours of the tag
def order(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

# Compute homography 
def homograph(p, p1):
    A = []
    p2 = order(p)

    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :] / Vh[-1, -1]
    h = np.reshape(l, (3, 3))
    
    return h

# Generate contours to detect corners of the tag
def contour_(frame):
    test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test_blur = cv2.GaussianBlur(test_img1, (5, 5), 0)
    
    edge = cv2.Canny(test_blur, 75, 200)
    
    edge1 = copy.copy(edge)
    contour_list = list()
    # Get countours and hierarchy
    cnts, h = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # We want the inner most contours using the RETR_TREE method
    index = list()
    for hier in h[0]:
        if hier[3] != -1:
            index.append(hier[3])

    # Approximation to create more whole contour image 
    for c in index:
        arc = cv2.arcLength(cnts[c], True)
        approx = cv2.approxPolyDP(cnts[c], 0.02 * arc, True)
        if len(approx) > 4:
            arc2 = cv2.arcLength(cnts[c - 1], True)
            corners = cv2.approxPolyDP(cnts[c - 1], 0.02 * arc2, True)
            contour_list.append(corners)   
            
    new_contour_list = list()
    for contour in contour_list:
        if len(contour) == 4:
            new_contour_list.append(contour)
            
    final_contour_list = list()
    for element in new_contour_list:
        if cv2.contourArea(element) < 2500:
            final_contour_list.append(element)
            
    return final_contour_list

# Rotate testudo based on the original orientation
def rotate(location, im):
    if location == "BR":
        return im
    elif location == "TR":
        im = cv2.flip(im, 0)
        return im
    elif location == "TL":
        im = cv2.flip(im, 1)
        im = cv2.flip(im, 0)
        return im
    elif location == "BL":
        im = cv2.flip(im, 1)
        return im

# Create video frame generated frames
# Leave commented out during console testing
# def visual(frames):
#     path = ''
#     img=[]
#     for i in range(0,frames):
#         img.append(cv2.imread(path+f'frame{i}.jpg'))
        
#     height,width,layers=img[1].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video=cv2.VideoWriter('Tag0_Solution.avi', fourcc, 15,(width,height))

#     for j in range(0,len(img)):
#         video.write(img[j])

#     cv2.destroyAllWindows()
#     video.release()
#     return

# Main function to generate image overlay
def solve(frame, p1, count, read, not_read):
    final_contour_list = contour_(frame)
    testu_list = list()
    # Track which history list to update
    contour_num = 1
    
    # Only move forward with usable corner points, the first three.
    if final_contour_list != []:
        for i in final_contour_list[:3]:
            cv2.drawContours(frame, i, -1, (0, 255, 0), 2)
            c_rez = i[:, 0]
            # Order corner points
            order_ = order(c_rez)
            # Call homograpghy
            H_matrix = homograph(p1, order_)
            # Call warp 
            tag = warped(frame, H_matrix, 200, 200, order_)
            
            # Show original frame and warped tag
            cv2.imshow("Outline", frame)
            # cv2.imwrite("outlined_frame.jpg", frame) 
            cv2.imshow("Tag after Homo", tag)
            # cv2.imwrite("Tag_homo.jpg", tag) 
            
            # Convert to gray scale to make averaging easier
            tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
            # Call code 
            tag_code, location = code(tag1, contour_num)
            print(location)
            empty = np.full(frame.shape, 0, dtype='uint8')
            # Orient the testud image
            if location != None:
                testu_flip = rotate(location, testu_resize)
                if tag_code != None:
                    print("Tag Code: " + str(tag_code))
                    # Call homograpghy
                    H_testu = homograph(order_, p1)
                    # Call warp
                    testu_overlap = testu_warped(testu_flip, H_testu, frame, order_)
                    # Ensure image warp isnt blank
                    if not np.array_equal(testu_overlap, empty):
                        testu_list.append(testu_overlap.copy())
            
            # Append correct code and location storage
            if contour_num == 1:
                loc_his1.append(location)
                code_his1.append(tag_code)
            if contour_num == 2:
                loc_his1.append(location)
                code_his1.append(tag_code)
            if contour_num == 3:
                loc_his1.append(location)
                code_his1.append(tag_code)
                
        contour_num += 1
    
    # Initialize mask                
    mask = np.full(frame.shape, 0, dtype='uint8')
    # Continue if testude was warped
    if testu_list != []:
        # Apply warped testudo to the mask
        for i in testu_list:
            temp = cv2.add(mask, i.copy())
            mask = temp
            
            testu_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            r, testu_bin = cv2.threshold(testu_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(testu_bin)
            mask_3d = frame.copy()
            mask_3d[:, :, 0] = mask_inv
            mask_3d[:, :, 1] = mask_inv
            mask_3d[:, :, 2] = mask_inv
            # Create final image of all testudo overlays
            img_masked = cv2.bitwise_and(frame, mask_3d)
            final_image = cv2.add(img_masked, mask)
            cv2.imwrite("mask.jpg", img_masked)
            
        final_resize = cv2.resize(final_image, (frame.shape[1]*2, frame.shape[0]*2))
        
        ###
        # Part of video creation, replace path variable with frame storage location
        # path = ''
        # cv2.imwrite((path+f'frame{count}.jpg'), final_resize)
        # count += 1
        # read += 1
        ###
        
        # Show final image
        cv2.imshow('Final', final_resize)
        # cv2.imwrite("Final.jpg", final_resize)
        cv2.waitKey(1)
    ###
    # Part of video creation, replace path variable with frame storage location    
    # else:
    #     path = ''
    #     cv2.imwrite((path+f'frame{count}.jpg'), frame)
    #     count += 1
    #     not_read +=1
    ###

    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()
        
    return count, read, not_read

###
# Part of video creation
count = 0
yes = 0
no_ = 0
###

# Read the input video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    # Break loop at end of video
    if success == False:
        break
    # Resize frame
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Call main function
    count, yes, no_ = solve(img, p1, count, yes, no_)

###
# Part of video creation
# print(count)
# print('Yes', yes)
# print('No', no_)
# visual(count)
###

cap.release()
cv2.destroyAllWindows()
