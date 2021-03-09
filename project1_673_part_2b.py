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
import time


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

start_time = time.time()
r_storage = []
t_storage = []

# Initialize the location history and code history lists
loc_his1 = ['']
loc_his2 = ['']
loc_his3 = ['']
code_his1 = ['']
code_his2 = ['']
code_his3 = ['']

dim = 200
p1 = np.array([
    [0, 0],
    [dim - 1, 0],
    [dim - 1, dim - 1],
    [0, dim - 1]], dtype="float32")


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

# Use the Calibration matrix to get the rotation and translation vectors
def projection(h):
    K = np.array(
        [[1406.08415449821, 0, 0], 
         [2.20679787308599, 1417.99930662800, 0], 
         [1014.13643417416, 566.347754321696, 1]]).T
    
    # initialize r and t vectors
    r, t = 0,0
    # Ensure no singular matrix
    if np.linalg.det(h) != 0:
        h = np.linalg.inv(h)
        # Get K^(-1)H
        b_new = np.dot(np.linalg.inv(K), h)
        # r1, r2, and t with lambda scalar 
        b1 = b_new[:, 0].reshape(3, 1)
        b2 = b_new[:, 1].reshape(3, 1)
        b3 = b_new[:, 2].reshape(3, 1)
        # Lambda
        L = (np.linalg.norm((np.linalg.inv(K)).dot(b1)) + np.linalg.norm((np.linalg.inv(K)).dot(b2)))/2
        # Divide by lambda
        r1 = b1/L
        r2 = b2/L
        t = b3/L
        # Cross product of r1 and r2
        r3 = np.cross(r1.T, r2.T).reshape(3, 1)
        # Mearge r1, r2, and r3 values
        r = np.concatenate((r1, r2, r3), axis=1)

    return r, t, K

# Draw the cube
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # Draw ground layer in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),10)

    # Connect bottom and top with vlue lines
    for i in range(4):
        for j in range(4,8):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # Draw top layer in red
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

###
# Part if video creation
# def visual(frames):
#     path = '/home/elliottmcg/Desktop/School/Second Semester/Perception_673/frames/'
#     img=[]
#     for i in range(0,frames):
#         img.append(cv2.imread(path+f'frame{i}.jpg'))
        
#     height,width,layers=img[1].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video=cv2.VideoWriter('Multi_cube_solution.avi', fourcc, 5,(width,height))

#     for j in range(0,len(img)):
#         video.write(img[j])

#     cv2.destroyAllWindows()
#     video.release()
#     return
###

# Boundry conditions for the r and t vectors
def fix_rt(r, t):
    check_r = 0
    check_t = 0
    if isinstance(r, int) == False:
        for i in r:
            for j in i:
                if -2 < j < 2:
                    check_r += 1
        for i in t:
            for j in i:
                if -100 < j < 100 or j > 20000 or j < -20000:
                    check_t += 1
                    
        if check_r == 9:
            r_storage.append(r)
        else:
            r = r_storage[-1]
            
        if check_t != 3:
            t_storage.append(t)
        else: 
            t = t_storage[-1]
        
    else:
        r = r_storage[-1]
        t = t_storage[-1]
        
    return r, t

# main function to process the tag and draw the cube        
def solve(frame, p1, a, count):
    final_contour_list = contour_(frame)
    cube_list = list()
    
    # Corners of cube in 3D space, match dim of warp
    axis = np.float32(
        [[0, 0, 0], 
         [0, 200, 0], 
         [200, 200, 0], 
         [200, 0, 0], 
         [0, 0, -200], 
         [0, 200, -200], 
         [200, 200, -200],
         [200, 0, -200]])
    
    # Track which history list to update
    contour_num = 1
    
    mask = np.full(frame.shape, 0, dtype='uint8')
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
            cv2.imshow("Tag after Homo", tag)
            
            # Convert to gray scale to make averaging easier
            tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
            # Call code 
            tag_code, location = code(tag1, contour_num)
            print(location)
            # Make sure the warp is readable before moving on
            if location != None:
                if tag_code != None:
                    print("Tag Code: " + str(tag_code))
                    code_his.append(tag_code)
                    # Get the r and t values
                    r, t, K = projection(H_matrix)
                    # Store the first r and t values then check
                    if a == 0:
                        r_storage.append(r)
                        t_storage.append(t)
                    else:
                        r, t = fix_rt(r, t)
                    # Get the bottom and top corner points
                    points, jac = cv2.projectPoints(axis, r, t, K, np.zeros((1, 4)))
                    # Draw the cube
                    img = draw(mask, points)
                    cube_list.append(img.copy())
        
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
        
        # If the cube list is not empy create the mask and overlay the cube on the image
        if cube_list != []: 
            for cube in cube_list:
                temp = cv2.add(mask, cube.copy())
                mask = temp
                cv2.imwrite("cube.jpg", cube)
                
                cube_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                r, cube_bin = cv2.threshold(cube_gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(cube_bin)
                mask_3d = frame.copy()
                mask_3d[:, :, 0] = mask_inv
                mask_3d[:, :, 1] = mask_inv
                mask_3d[:, :, 2] = mask_inv
                # Create final image of all cube overlays
                img_masked = cv2.bitwise_and(frame, mask_3d)
                # cv2.imwrite("cube_mask.jpg", img_masked)
                final_image = cv2.add(img_masked, mask)
            
            
            final_resize = cv2.resize(final_image, (frame.shape[1]*2, frame.shape[0]*2))
            cv2.imshow("cube", final_resize)
            cv2.imwrite("final_cube.jpg", final_resize)
            ###
            # Part of video creation, replace path variable with frame storage location
            # path = ''
            # cv2.imwrite((path+f'frame{count}.jpg'), final_resize)
            # count += 1
            ###
            cv2.waitKey(1)
    

    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()
        
    return count
        
a = 0
count = 0
# Read the input video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    if success == False:
        break
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    count = solve(img, p1, a, count)
    a += 1
    
    
###
# Part of video creation   
# visual(count)
###
cap.release()
cv2.destroyAllWindows()
