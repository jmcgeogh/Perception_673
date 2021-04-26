#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:50:59 2021

@author: elliottmcg
"""

import cv2
import numpy as np
import math
import sys
import matplotlib as plt
import matplotlib.cm as cmplt
from matplotlib import pyplot as pyplt

print('Please select which set of images you would like to test')
print('For data set 1, press 1')
print('For data set 2, press 2')
print('For data set 3, press 3')
data_select = int(input('Make Your Selection: '))

if data_select == 1:
    im0 = cv2.imread('im0_ds1.png')
    im1 = cv2.imread('im1_ds1.png')
    K0 = np.array(
        [[5299.313, 0, 1263.818],
         [0, 5299.313, 977.763],
         [0, 0, 1]])
    K1 = np.array(
        [[5299.313, 0, 1438.004],
         [0, 5299.313, 977.763],
         [0, 0, 1]])
    D = 174.186
    base = 177.288
    focal = 5299.313
    max_offset = 60
elif data_select == 2:
    im0 = cv2.imread('im0_ds2.png')
    im1 = cv2.imread('im1_ds2.png')
    K0 = np.array(
        [[4396.869, 0, 1353.072],
         [0, 4396.869, 989.702],
         [0, 0, 1]])
    K1 = np.array(
        [[4396.869, 0, 1538.86],
         [0, 4396.869, 989.702],
         [0, 0, 1]])
    D = 185.788
    base = 144.049
    focal = 4396.869
    max_offset = 30
elif data_select == 3:
    im0 = cv2.imread('im0_ds3.png')
    im1 = cv2.imread('im1_ds3.png')
    K0 = np.array(
        [[5806.559, 0, 1429.219],
         [0, 5806.559, 993.403],
         [0, 0, 1]])
    K1 = np.array(
        [[5806.559, 0, 1543.51],
         [0, 5806.559, 993.403],
         [0, 0, 1]])
    D = 114.291
    base = 174.019
    focal = 5806.559
    max_offset = 30
else:
    print('Incorrect input')
    print('Exiting...')
    sys.exit()
   
def scale(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

im0 = scale(im0, 25)
im1 = scale(im1, 25)

def keypoints(img0, img1):
    orb = cv2.ORB_create()

    kp1 = orb.detect(img0,None)
    kp2 = orb.detect(img1,None)

    kp1, des1 = orb.compute(img0, kp1)
    kp2, des2 = orb.compute(img1, kp2)

    key1 = cv2.drawKeypoints(img0, kp1, None, color=(0,255,0), flags=0)
    key2 = cv2.drawKeypoints(img1, kp2, None, color=(0,255,0), flags=0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    pts1 = []
    pts2 = []
    for mat in matches[:100]:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        pts1.append((x1, y1))
        pts2.append((x2, y2))

    img4 = cv2.drawMatches(img0,kp1,img1,kp2,matches[:len(pts1)],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return pts1, pts2, img4, key1, key2

def fundamental_matrix(pts1, pts2):

    n = len(pts1)
    A = np.zeros((n, 9), dtype="float32")
    for i in range(n):
        x1 = pts1[i][0]
        x2 = pts2[i][0]
        y1 = pts1[i][1]
        y2 = pts2[i][1]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        
    u, s, vh = np.linalg.svd(A)    
    smallest = list(s).index(min(s))
    fmatrix = vh[:,smallest]   
    fmatrix = fmatrix.reshape((3, 3))    
    # print('rank', np.linalg.matrix_rank(fmatrix))
    
    uf, sf, vfh = np.linalg.svd(fmatrix)
    smallest_f = list(sf).index(min(sf))
    sf[smallest_f] = 0
    sf = np.diag(sf)
    f_reduced = np.matmul(np.matmul(uf, sf), vfh)
    # print('rank', np.linalg.matrix_rank(f_reduced))
    
    return f_reduced

def essential_matrix(f_matrix, K0, K1):
    
    e_matrix = np.matmul(np.matmul(K1.T, f_matrix), K0)
    
    return e_matrix

def camera_pose(e_matrix, K):
    
    u, s, vh = np.linalg.svd(e_matrix)
    W = np.array(
        [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]])
    
    c1 = u[:,2]
    C1 = np.array([[c1[0], c1[1], c1[2]]])
    c2 = -u[:,2]
    C2 = np.array([[c2[0], c2[1], c2[2]]])
    c3 = u[:,2]
    C3 = np.array([[c3[0], c3[1], c3[2]]])
    c4 = -u[:,2]
    C4 = np.array([[c4[0], c4[1], c4[2]]])
    
    R1 = np.matmul(np.matmul(u, W), vh)
    R2 = np.matmul(np.matmul(u, W), vh)
    R3 = np.matmul(np.matmul(u, W.T), vh)
    R4 = np.matmul(np.matmul(u, W.T), vh)
    
    R_list = [R1, R2, R3, R4]
    C_list = [C1, C2, C3, C4]
    R_correct = []
    C_correct = []
    n = 0
    for i in R_list:
        if np.linalg.det(i) < 0:
            R_correct.append(-i)
            C_correct.append(-C_list[n])
        else:
            R_correct.append(i)
            C_correct.append(C_list[n])
        n += 1

    I = np.array([[1,0,0], [0,1,1], [0,0,1]])

    I1 = np.append(I, C_correct[0].T, axis=1)
    I2 = np.append(I, C_correct[1].T, axis=1)
    I3 = np.append(I, C_correct[2].T, axis=1)
    I4 = np.append(I, C_correct[3].T, axis=1)

    P1 = np.matmul(np.matmul(K, R_correct[0]), I1)
    P2 = np.matmul(np.matmul(K, R_correct[1]), I2)
    P3 = np.matmul(np.matmul(K, R_correct[2]), I3)
    P4 = np.matmul(np.matmul(K, R_correct[3]), I4)
    
    P = [P1, P2, P3, P4]
    
    return P, R_correct, C_correct

def triangulation(pts1, pts2, P):
    
    X = []
    for i in range(len(pts1)):
        x1 = pts1[i][0]
        y1 = pts1[i][1]
        x2 = pts2[i][0]
        y2 = pts2[i][1]
        A1 = y1*P[0][2] - P[0][1]
        A2 = x1*P[0][2] - P[0][0]
        A3 = y2*P[1][2] - P[1][1]
        A4 = x2*P[1][2] - P[1][0]
        A = [A1, A2, A3, A4]
        
        u, s, vh = np.linalg.svd(A)
        X.append(vh[3])
    
    return X

def find_pose(Rs, Cs, X):

    check1 = []
    check2 = []
    check3 = []
    check4 = []
    for i in range(len(Cs)):
        for j in X:
            check = Rs[i][:, 2]*(j[:3]-Cs[i])
            # print(check[0])
            if check[0][0] > 0 and check[0][1] > 0 and check[0][2] > 0:
                if i == 0:
                    check1.append(1)
                if i == 1:
                    check2.append(1)
                if i == 2:
                    check3.append(1)
                if i == 3:
                    check4.append(1)
                    
    checks = [len(check1), len(check2), len(check3), len(check4)]
    max_index = checks.index(max(checks))
    R = Rs[max_index]
    T = Cs[max_index]
    
    return R, T

def ssd(rect_img1, rect_img2):
      
    gray1 = cv2.cvtColor(rect_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect_img2, cv2.COLOR_BGR2GRAY)
    rows, cols = gray1.shape[:2]  
    
    disp = np.zeros((cols, rows), np.uint8)
    disp.shape = rows, cols
    window = 6
       
    w_half = int(window / 2)    
    adjust = 255 / max_offset 
    adj_val = []
    for y in range(w_half, rows - w_half):      
        print('rows left', (rows-w_half)-y)    
        for x in range(w_half, cols - w_half):
            best_offset = 0
            prev_ssd = None
            for offset in range(max_offset):               
                ssd = 0
                diff = 0                                            
                # window search
                for v in range(-w_half, w_half):
                    for u in range(-w_half, w_half):
                        diff = int(gray1[y+v, x+u]) - int(gray2[y+v, (x+u) - offset])  
                        ssd += diff * diff              
                
                # Continue to check for the smallest ssd value
                if prev_ssd == None:
                    prev_ssd = ssd
                    best_offset = offset
                elif ssd < prev_ssd: 
                    prev_ssd = ssd
                    best_offset = offset
                            
            disp[y, x] = int(best_offset * adjust)
            adj_val.append(best_offset * adjust)
    
    norm_disp = cv2.normalize(disp, disp, alpha=255,
                               beta=0, norm_type=cv2.NORM_MINMAX)
    # print(max(adj_val))
    # norm_val = []
    # for y in range(rows):
    #     for x in range(cols):
    #         norm_val.append(norm_disp[y,x])
    
    # print('after norm', max(norm_val))      
    return norm_disp


def depth(image):
    
    rows, cols = image.shape[:2]  
    
    depth_image = np.zeros((cols, rows), np.uint8)
    depth_image.shape = rows, cols
    Z_val = []
    for y in range(rows):
        for x in range(cols):
            if image[y,x] == 0:
                Z_val.append(0)
                depth_image[y, x] = 0
            else:
                Z = (focal*base)/image[y,x]
                Z_val.append(Z)
                depth_image[y, x] = Z
    
    norm_depth = cv2.normalize(depth_image, depth_image, alpha=255,
                               beta=0, norm_type=cv2.NORM_MINMAX)
    
    # print('before norm', max(Z_val))
    # norm_val = []
    # for y in range(rows):
    #     for x in range(cols):
    #         norm_val.append(norm_depth[y,x])
    
    # print('after norm', max(norm_val))
    return norm_depth

#####Part 1#####
pts1, pts2, match, key1, key2 = keypoints(im0, im1)
cv2.imshow('Left KeyPoints', key1)
# cv2.imwrite('Left_KeyPoints.jpg', key1)
cv2.waitKey(0)
cv2.imshow('Right KeyPoints', key2)
# cv2.imwrite('Right_KeyPoints.jpg', key2)
cv2.waitKey(0)
cv2.imshow('Original Matching Points', match)
# cv2.imwrite('Original_Matching_Points.jpg', match)
cv2.waitKey(0)

f_matrix = fundamental_matrix(pts1, pts2)
e_matrix = essential_matrix(f_matrix, K0, K1)
P, R_list, C_list = camera_pose(e_matrix, K0)
X = triangulation(pts1, pts2, P)
R, T = find_pose(R_list, C_list, X)


#####Part 2#####
method=cv2.FM_RANSAC
fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
        )
pts1 = np.array(pts1)
pts2 = np.array(pts2)
h1, w1 = im0.shape[:2]
h2, w2 = im1.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, fundamental_matrix, imgSize=(w1, h1))

img1_rectified = cv2.warpPerspective(im0, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(im1, H2, (w2, h2))
cv2.imshow("Left Rectified ", img1_rectified)
cv2.imshow("Right Rectified", img2_rectified)
# cv2.imwrite('rect1.jpg', img1_rectified)
# cv2.imwrite('rect2.jpg', img2_rectified)
cv2.waitKey(0)

rect_pts1, rect_pts2, rect_match, rect_key1, rect_key2 = keypoints(img1_rectified, img2_rectified)
cv2.imshow('Warped Left KeyPoints', rect_key1)
# cv2.imwrite('Warped_Left_KeyPoints.jpg', rect_key1)
cv2.waitKey(0)
cv2.imshow('Warped Right KeyPoints', rect_key2)
# cv2.imwrite('Warped_Right_KeyPoints.jpg', rect_key2)
cv2.waitKey(0)
cv2.imshow('Warped Matching Points', rect_match)
# cv2.imwrite('Warped_Matching_Points.jpg', rect_match)
cv2.waitKey(0)

#####Part 3#####
ssd_image = ssd(img1_rectified, img2_rectified)
cv2.imshow('SSD Disparity Image', ssd_image)
# cv2.imwrite('ssd.jpg', ssd_image)
# cv2.waitKey(0)

heatmap = cv2.applyColorMap(ssd_image, colormap=cv2.COLORMAP_JET)
cv2.imshow('Disparity Heatmap', heatmap)
# cv2.imwrite('heatmap.jpg', heatmap)
# cv2.waitKey(0)

#####Part 4##### 
# imgL = cv2.imread('rect1.jpg',0)
# imgR = cv2.imread('rect2.jpg',0)
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL,imgR)
# pyplt.figure(1)
# pyplt.imshow(disparity,'gray')
# pyplt.figure(2)
# pyplt.imshow(disparity,'plasma')
# pyplt.show()
depth_image = depth(ssd_image)
cv2.imshow('Depth', depth_image)
# cv2.imwrite('depth.jpg', depth_image)
cv2.waitKey(0)
depth_heatmap = cv2.applyColorMap(depth_image, colormap=cv2.COLORMAP_JET)
cv2.imshow('Depth Heatmap', depth_heatmap)
# cv2.imwrite('depth_heatmap.jpg', depth_heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()