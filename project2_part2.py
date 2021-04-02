#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:46:05 2021

@author: elliottmcg
"""
import numpy as np
import cv2
import copy
from scipy import ndimage
from scipy.signal import find_peaks
import math
import matplotlib.pyplot as plt

# # Create video from data set 1
# # Leave commented out during console testing
# def visual(frames = 303):
#     path = '/home/elliottmcg/Desktop/School/Second Semester/Perception_673/Project2/data1/'
#     img=[]
#     for i in range(0,frames):
#         if i < 10:
#             img.append(cv2.imread(path+f'000000000{i}.png'))
#         if i >= 10 and i < 100:
#             img.append(cv2.imread(path+f'00000000{i}.png'))
#         if i >= 100:
#             img.append(cv2.imread(path+f'0000000{i}.png'))
        
#     height,width,layers=img[1].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video=cv2.VideoWriter('data_set1_video.avi', fourcc, 15,(width,height))

#     for j in range(0,len(img)):
#         video.write(img[j])

#     cv2.destroyAllWindows()
#     video.release()
#     return

# visual()

# Select video to test with
print("Choose from the selected options for Tag videos")
print("press 1 for data_set1_video")
print("press 2 for challenge_video")
print("")
a = int(input("Make your selection: "))
if a == 1:
    cap = cv2.VideoCapture('data_set1_video.avi')
    hcap = cv2.VideoCapture('data_set1_video.avi')
    # Camera Matrix
    K = np.array(
        [[903.7596, 0, 695.7519],
         [0, 901.9653, 224.2509],
         [0, 0, 1]])
    # Distortion Coefficients
    D = np.array([[-0.3639558, 0.1788651, 0.0006029694, -.0003922424, -.05382460]])
elif a == 2:
    cap = cv2.VideoCapture('challenge_video.mp4')
    hcap = cv2.VideoCapture('challenge_video.mp4')
    # Camera Matrix
    K = np.array(
        [[1154.22732, 0, 671.627794],
         [0, 1148.18221, 386.046312],
         [0, 0, 1]])
    # Distortion Coefficients
    D = np.array([[-.242565104, -.0477893070, -.00131388084, -.0000879107779, .0220573263]])
else:
    print("sorry selection could not be identified, exiting code")
    exit(0)


def canny1(image):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0) 
    
    # Use Median Blur
    blur = cv2.medianBlur(gray, 5)
    # cv2.imshow("blur", blur)
    # cv2.waitKey(0)
    
    # Threshold the image for the right and left lane 
    ret,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY)
    thresh[:, 940:] = 0
    thresh[:, :200] = 0
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0) 
    
    # Use canny edge detection
    edges = cv2.Canny(thresh,250,400)
    
    return edges

def canny2(image):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0) 
    
    # Used Median Blur
    blur = cv2.medianBlur(gray, 5)
    # cv2.imshow("blur", blur)
    # cv2.waitKey(0)
    
    # Threshold the right and left lane
    ret1,thresh1 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(blur,151,255,cv2.THRESH_BINARY)
    thresh2[:, 565:] = 0
    thresh1[:, 1100:] = 0
    thresh = thresh2 + thresh1
    # cv2.imshow("thresh1", thresh1)
    # cv2.waitKey(0) 
    # cv2.imshow("thresh2", thresh2)
    # cv2.waitKey(0) 
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0) 
    # cv2.imwrite("Thresh.jpg", thresh)
    
    # Use canny edge detection
    edges = cv2.Canny(thresh,250,400)
    
    return edges

def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    h = width//2
    
    # Create empty ROI 
    if a == 1:
        roi_ = np.array([[(260, height), (850, height), (h, 0)]])
    if a == 2:
        roi_ = np.array([[(250, height), (1200, height), (h, 0)]])
        
    # Fill the ROI with white pixels
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi_, 255) 
    # cv2.imwrite("ROI mask.jpg", mask)
    
    # Get the region within the ROI
    masked_image = cv2.bitwise_and(image, mask) 
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.imwrite("ROI.jpg", masked_image)
    
    return masked_image

def order(pts):
    
    # Order the corner locations in accordance with x value
    rect = np.zeros((4, 2), dtype="float32")
    x = []
    y = []
    for points in pts:
        x.append(points[0])
        y.append(points[1])
        
    for i in range(len(pts)):
        index = x.index(min(x))
        rect[i] = [x.pop(index), y.pop(index)]
    
    return rect

# Warp points for data set 1 video
if a == 1:
    dim = 200
    p1 = np.array([
        [dim, dim*3],
        [dim, dim*2.6],
        [dim*2, dim],
        [dim*2, dim*2.8]], dtype="float32")

# Warp points for challenge video
if a == 2:
    dim = 200
    p1 = np.array([
        [dim, dim*3],
        [dim, dim],
        [dim*2, dim],
        [dim*2, dim*2.7]], dtype="float32")
        
def find_homo(edges, frame):
    
    # Get the image only within ROI
    roi_ = roi(edges)
    
    # Get the corner points 
    corners = cv2.goodFeaturesToTrack(roi_,4,.01,100)
    corners = np.int0(corners)
    corner_loc = []

    # Display corner ditection and store coordinates
    for i in corners:
        x,y = i.ravel()
        cv2.circle(roi_,(x,y),10,150,-1)
        corner_loc.append([x,y])
    
    # Order the corners
    lane_corners = order(corner_loc)
    # cv2.imshow('roi', roi_)
    # cv2.waitKey(0)
    # cv2.imwrite("Corners.jpg", roi_)
    
    # Get the homography of the frame
    h_matrix, _ = cv2.findHomography(lane_corners, p1)
    # print(h_matrix)
    
    return h_matrix, lane_corners

def return_image(lane_image, lane_corners, original, direction, count):
    
    # Unwarp the image
    x = original.shape[1]
    y = original.shape[0]
    h_matrix, _ = cv2.findHomography(p1, lane_corners)
    final = cv2.warpPerspective(lane_image,h_matrix, (x,y))
    # cv2.imwrite("final warp.jpg", final)
    
    # Mesh the original image with the unwarped final image
    result = cv2.addWeighted(final, .4, original, 1, 0)
    
    # Overlay the road condition onto the image
    if direction == 'R':
        if a == 1:
            h = y - 20
            w = x//3
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Road Turns Right', (w,h), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if a == 2:
            h = y - 50
            w = int(x/2.5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Road Turns Right', (w,h), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if direction == 'L':
        if a == 1:
            h = y - 20
            w = x//3
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Road Turns Left', (w,h), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if a == 2:
            h = y - 50
            w = int(x/2.5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Road Turns Left', (w,h), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if direction == 'S':
        if a == 1:
            h = y - 20
            w = x//3
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Continue Straight', (w,h), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if a == 2:
            h = y - 50
            w = int(x/2.5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Continue Straight', (w,h), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    
    # cv2.imshow('final', result)
    # cv2.waitKey(0)
    # cv2.imwrite("final.jpg", result)
    
    ###
    # Part of video creation, replace path variable with frame storage location
    path = '/home/elliottmcg/Desktop/School/Second Semester/Perception_673/frames/'
    cv2.imwrite((path+f'frame{count}.jpg'), result)
    count += 1
    ###
    
    return count

# Create storage lists
prev_peak = []
prev_pts1 = []
prev_pts2 = []
def histogram1(warp):
    
    # Grayscale the warp and threshold
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(warp_gray,240,255,cv2.THRESH_BINARY)
    # cv2.imshow('warp gray thresh', thresh)
    # cv2.waitKey(1)
    
    # Create histogram of the sum of y values over an x coordinate
    histogram = np.sum(thresh, axis=0)
    
    # Find the target peaks within the range of the lane warp
    target_peaks = []
    x_peaks, _ = find_peaks(histogram, height=1000)
    x_peaks = list(x_peaks)
    y_peaks = list(histogram[x_peaks])
    if y_peaks != []:
        while target_peaks == []:
            if y_peaks == []:
                target_peaks = prev_peak[-1]
            index = y_peaks.index(max(y_peaks))
            if 150 < x_peaks[index] < 450:
                target_peaks.append(x_peaks.pop(index))
                y_peaks.pop(index)
            else:
                x_peaks.pop(index)
                y_peaks.pop(index)
    else:
        target_peaks = prev_peak[-1]
    
    
    while len(target_peaks) < 2:
        if y_peaks != []:
            index = y_peaks.index(max(y_peaks))
            if (target_peaks[0]-50) > x_peaks[index] > 150 or 450 > x_peaks[index] > (target_peaks[0]+50):
                target_peaks.append(x_peaks.pop(index))
            else:
                y_peaks.pop(index)
                x_peaks.pop(index) 
        else:
            target_peaks = prev_peak[-1]
    
    # Store the target peaks
    prev_peak.append(target_peaks)
    
    # Get the top values
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    tops = np.zeros((600,600,3), np.uint8)
    for i in range(6):
        x1.append(target_peaks[0] + i)
        x2.append(target_peaks[1] + i)
    for i in range(1, 6):
        x1.append(target_peaks[0] - i)
        x2.append(target_peaks[1] - i)
        
    x1.sort()
    x2.sort()
    x1_targ = []
    x2_targ = []
    # print(x1)
    for x in x1:
        y_count = 0
        y_sum = 0
        for y in range(600):    
            if thresh[y][x] > 0:
                tops[y][x] = (255,0,0)
                y_count += 1
                y_sum += y
    
        if y_count > 0:
            y1.append(y_sum//y_count)
            x1_targ.append(x)
            
    for x in x2:
        y_count = 0
        y_sum = 0
        for y in range(600):    
            if thresh[y][x] > 0:
                tops[y][x] = (255,0,0)
                y_count += 1
                y_sum += y
        
        if y_count > 0:
            y2.append(y_sum//y_count)
            x2_targ.append(x)
            
    # Get the lane lines from the max and min y values
    if len(y1) >= 2:
        max1_index = y1.index(max(y1))
        max_y1 = y1[max1_index]
        max_x1 = x1_targ[max1_index]
        
        min1_index = y1.index(min(y1))
        min_y1 = y1[min1_index]
        min_x1 = x1_targ[min1_index]
        m = (min_y1 - max_y1)/(min_x1 - max_x1)
        b = min_y1 - (m*min_x1)
        start_x1 = int((600-b)/m)
        end_x1 = int((0-b)/m)
        # print('start 1', start_x1)
        # print('end 1', end_x1)
        if 0 < start_x1 < 600 and 0 < end_x1 < 600:
            sp_1 = (start_x1, 600)
            ep_1 = (end_x1, 0)
            prev_pts1.append([sp_1, ep_1])
        else:
            prev = prev_pts1[-1]
            sp_1 = prev[0]
            ep_1 = prev[1]  
    else:
        prev = prev_pts1[-1]
        sp_1 = prev[0]
        ep_1 = prev[1]
    
    if len(y2) >= 2:
        max2_index = y2.index(max(y2))
        max_y2 = y2[max2_index]
        max_x2 = x2_targ[max2_index]
        
        min2_index = y2.index(min(y2))
        min_y2 = y2[min2_index]
        min_x2 = x2_targ[min2_index]
        m = (min_y2 - max_y2)/(min_x2 - max_x2)
        b = min_y2 - (m*min_x2)
        start_x2 = int((600-b)/m)
        end_x2 = int((0-b)/m)
        # print('start 2', start_x2)
        # print('end 2', end_x2)
        if 0 < start_x2 < 600 and 0 < end_x2 < 600:
            sp_2 = (start_x2, 600)
            ep_2 = (end_x2, 0)
            prev_pts2.append([sp_2, ep_2])
        else:
            prev = prev_pts2[-1]
            sp_2 = prev[0]
            ep_2 = prev[1] 
    else:
        prev = prev_pts2[-1]
        sp_2 = prev[0]
        ep_2 = prev[1]
    
    # Draw the expected lane lines
    cv2.line(tops, sp_1, ep_1, (0,0,255), 1)
    cv2.line(tops, sp_2, ep_2, (0,0,255), 1)
    # print('line 1', sp_1, ep_1)
    # print('line 2', sp_2, ep_2)
    
    # cv2.imshow('tops', tops)
    # cv2.waitKey(1)
    
            
    # cv2.imshow('tops', tops)
    # cv2.waitKey(1)
    # print(len(histogram))
    
    # Plot the histogram with peaks
    # plt.plot(histogram)
    # plt.plot(target_peaks, histogram[target_peaks], "x")
    # plt.show()
    
    # Output the lane line points
    line_1_pts = [sp_1, ep_1]
    line_2_pts = [sp_2, ep_2]
    return target_peaks, tops, line_1_pts, line_2_pts

def histogram2(warp):
    
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('warp gray', warp_gray)
    # cv2.waitKey(0)
    
    ret,thresh = cv2.threshold(warp_gray,200,255,cv2.THRESH_BINARY)
    # cv2.imshow('warp threah', thresh)
    # cv2.waitKey(0)
    # cv2.imwrite("warp_thresh.jpg", thresh)
    
    # Convert the image to HSV
    hsv = cv2.cvtColor(warp,cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)
    # cv2.waitKey(0)
    
    # Thresh the yellow line in the HSV image
    yellow_lower = np.array([20, 55, 145])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_output = cv2.bitwise_and(warp, warp, mask=mask_yellow)
    yellow_gray = cv2.cvtColor(yellow_output, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("warp_yellow.jpg", yellow_output)
    
    # cv2.imshow('color threah', yellow_output)
    # cv2.waitKey(1)
    # cv2.imshow('yellow gray', yellow_gray)
    # cv2.waitKey(1)
    
    # Combine the HSV thresh and gray thresh
    combine = yellow_gray + thresh
    # cv2.imwrite("combine.jpg", combine)
    # cv2.imshow('combine', combine)
    # cv2.waitKey(1)
    
    # Histogram and line detection same as histogram1
    histogram = np.sum(combine, axis=0)
    target_peaks = []
    x_peaks, _ = find_peaks(histogram, height=1000)
    x_peaks = list(x_peaks)
    y_peaks = list(histogram[x_peaks])
    if y_peaks != []:
        while target_peaks == []:
            if y_peaks == []:
                target_peaks = prev_peak[-1]
            index = y_peaks.index(max(y_peaks))
            if 150 < x_peaks[index] < 450:
                target_peaks.append(x_peaks.pop(index))
                y_peaks.pop(index)
            else:
                x_peaks.pop(index)
                y_peaks.pop(index)
    else:
        target_peaks = prev_peak[-1]
    
    # print(target_peaks)
    while len(target_peaks) < 2:
        if y_peaks != []:
            index = y_peaks.index(max(y_peaks))
            if (target_peaks[0]-50) > x_peaks[index] > 150 or 450 > x_peaks[index] > (target_peaks[0]+50):
                target_peaks.append(x_peaks.pop(index))
            else:
                y_peaks.pop(index)
                x_peaks.pop(index) 
        else:
            target_peaks = prev_peak[-1]
    
    prev_peak.append(target_peaks)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    tops = np.zeros((600,600,3), np.uint8)
    for i in range(6):
        x1.append(target_peaks[0] + i)
        x2.append(target_peaks[1] + i)
    for i in range(1, 6):
        x1.append(target_peaks[0] - i)
        x2.append(target_peaks[1] - i)
        
    x1.sort()
    x2.sort()
    x1_targ = []
    x2_targ = []            
    # print(x1)
    for x in x1:
        y_count = 0
        y_sum = 0
        for y in range(600):    
            if combine[y][x] > 0:
                tops[y][x] = (255,0,0)
                y_count += 1
                y_sum += y
    
        if y_count > 0:
            y1.append(y_sum//y_count)
            x1_targ.append(x)
    
            
    for x in x2:
        y_count = 0
        y_sum = 0
        for y in range(600):    
            if combine[y][x] > 0:
                tops[y][x] = (255,0,0)
                y_count += 1
                y_sum += y
        
        if y_count > 0:
            y2.append(y_sum//y_count)
            x2_targ.append(x)
            
    # cv2.imwrite("tops.jpg", tops)       
    if len(y1) >= 2:
        max1_index = y1.index(max(y1))
        max_y1 = y1[max1_index]
        max_x1 = x1_targ[max1_index]
        
        min1_index = y1.index(min(y1))
        min_y1 = y1[min1_index]
        min_x1 = x1_targ[min1_index]
        m = (min_y1 - max_y1)/(min_x1 - max_x1)
        b = min_y1 - (m*min_x1)
        start_x1 = int((600-b)/m)
        end_x1 = int((0-b)/m)
        # print('start 1', start_x1)
        # print('end 1', end_x1)
        if 0 < start_x1 < 600 and 0 < end_x1 < 600:
            sp_1 = (start_x1, 600)
            ep_1 = (end_x1, 0)
            prev_pts1.append([sp_1, ep_1])
        else:
            prev = prev_pts1[-1]
            sp_1 = prev[0]
            ep_1 = prev[1]  
    else:
        prev = prev_pts1[-1]
        sp_1 = prev[0]
        ep_1 = prev[1]
    
    if len(y2) >= 2:
        max2_index = y2.index(max(y2))
        max_y2 = y2[max2_index]
        max_x2 = x2_targ[max2_index]
        
        min2_index = y2.index(min(y2))
        min_y2 = y2[min2_index]
        min_x2 = x2_targ[min2_index]
        m = (min_y2 - max_y2)/(min_x2 - max_x2)
        b = min_y2 - (m*min_x2)
        start_x2 = int((600-b)/m)
        end_x2 = int((0-b)/m)
        # print('start 2', start_x2)
        # print('end 2', end_x2)
        if 0 < start_x2 < 600 and 0 < end_x2 < 600:
            sp_2 = (start_x2, 600)
            ep_2 = (end_x2, 0)
            prev_pts2.append([sp_2, ep_2])
        else:
            prev = prev_pts2[-1]
            sp_2 = prev[0]
            ep_2 = prev[1] 
    else:
        prev = prev_pts2[-1]
        sp_2 = prev[0]
        ep_2 = prev[1]
    
    cv2.line(tops, sp_1, ep_1, (0,0,255), 2)
    cv2.line(tops, sp_2, ep_2, (0,0,255), 2)
    # cv2.imwrite("tops_red.jpg", tops)
    
    # cv2.imshow('tops', tops)
    # cv2.waitKey(0)
    line_1_pts = [sp_1, ep_1]
    line_2_pts = [sp_2, ep_2]
    
    # print(len(histogram))
    # plt.plot(histogram)
    # plt.plot(target_peaks, histogram[target_peaks], "x")
    # plt.show()
    
    return target_peaks, tops, line_1_pts, line_2_pts

# Store direction
dir_storage = []
def draw_lines(warp_lines, peaks, tops, line_1_pts, line_2_pts):
    
    # Sort the peaks
    peaks.sort()
    
    # Put the line detection endpoint in the correct order
    line1_list = []
    line2_list = []
    for i in range(2):
        for j in range(2):
            line1_list.append(line_1_pts[i][j])
            line2_list.append(line_2_pts[i][j])
            
    line1_index = line1_list.index(0)
    line2_index = line2_list.index(0)
    line1_point = line1_list[line1_index-1]
    line2_point = line2_list[line2_index-1]
    lines_points = [line1_point, line2_point]
    lines_points.sort()
    
    # Create the buffer for the right and left lanes
    buf = 10
    left_sp = (peaks[0], 0)
    left_ep = (peaks[0], 600)
    LLL_buf = peaks[0] - buf
    RLL_buf = peaks[0] + buf
    LLL_sp = (LLL_buf, 0)
    LLL_ep = (LLL_buf, 600)
    RLL_sp = (RLL_buf, 0)
    RLL_ep = (RLL_buf, 600)
    
    LRL_buf = peaks[1] - buf
    RRL_buf = peaks[1] + buf
    right_sp = (peaks[1], 0)
    right_ep = (peaks[1], 600)
    LRL_sp = (LRL_buf, 0)
    LRL_ep = (LRL_buf, 600)
    RRL_sp = (RRL_buf, 0)
    RRL_ep = (RRL_buf, 600)
    
    # Get the center of the lane
    average = (peaks[1]+peaks[0])//2
    average_sp = (average, 0)
    average_ep = (average, 600)
    
    # Direction conditions
    dir_ = 'S'
    if LLL_buf < lines_points[0] < RLL_buf and LRL_buf < lines_points[1] < RRL_buf:
        dir_ = 'S'
    if LLL_buf > lines_points[0] and LRL_buf > lines_points[1]:
        dir_ = 'L'
    if lines_points[0] > RLL_buf and lines_points[1] > RRL_buf:
        dir_ = 'R'
    if LLL_buf < lines_points[0] < RLL_buf and LRL_buf > lines_points[1]:
        dir_ = 'L'
    if LLL_buf < lines_points[0] < RLL_buf and lines_points[1] > RRL_buf:
        dir_ = 'R'
    if lines_points[0] > RLL_buf and LRL_buf < lines_points[1] < RRL_buf:
        dir_ = 'R'
    if LLL_buf > lines_points[0] and LRL_buf < lines_points[1] < RRL_buf:
        dir_ = 'L'
     
    # Store direction
    dir_storage.append(dir_)

    # Draw the lines on the warped image and the tops image    
    cv2.line(warp_lines, left_sp, left_ep, (0,255,0), 5)
    cv2.line(warp_lines, right_sp, right_ep, (0,255,0), 5)
    # cv2.imwrite("warp lines.jpg", warp_lines)

    cv2.line(tops, LLL_sp, LLL_ep, (0,255,0), 2)
    cv2.line(tops, RLL_sp, RLL_ep, (0,255,0), 2)
    cv2.line(tops, LRL_sp, LRL_ep, (0,255,0), 2)
    cv2.line(tops, RRL_sp, RRL_ep, (0,255,0), 2)
    
    cv2.line(tops, average_sp, average_ep, (0,255,0), 2)
    
    # cv2.imwrite("tops_buff.jpg", tops)
    
    # Create the lane area and overaly it onto the warped image
    clean_warp = warp_lines.copy()
    final = warp_lines.copy()
    cv2.rectangle(clean_warp, left_sp, right_ep, (255, 0, 0), -1)
    cv2.addWeighted(clean_warp, .5, final, .5, 0, final)
    # cv2.imwrite("warp lines.jpg", final)
    # cv2.imshow('warp lines', final)
    # cv2.waitKey(1)
    
    # cv2.imshow('top lines', tops)
    # cv2.waitKey(0)
    # print(dir_)
    
    return final, dir_

def undistort(img):
    
    # Getting the new optimal camera matrix
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K,D,(w,h),1, (w,h))
    
    # Undistorting
    undst = cv2.undistort(img, K, D, None, new_K)
    
    # Cropping the image
    x,y,w,h = roi
    undst = undst[y:+y+h, x:x+w]
    # cv2.imshow('undst',undst)
    # cv2.waitKey(1)
    # cv2.imwrite("Undist.jpg", undst)
    
    # cv2.imshow('original',img)
    # cv2.waitKey(1)
    # cv2.imwrite("Original.jpg", img)
    
    return undst

# Get the frames to perform Homograpghy on
if a == 1:
    hcap.set(1, 0)
    ret, frame = hcap.read()
    undst = undistort(frame)
    y = int(undst.shape[0])
    h = y//3
    crop_frame = undst[h+h//3:y, :]
    # cv2.imshow("cropped", crop_frame)
    # cv2.waitKey(0) 
    
    edges = canny1(crop_frame)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)
    
if a == 2:
    hcap.set(1, 457)
    ret, frame = hcap.read()
    undst = undistort(frame)
    y = int(undst.shape[0])
    h = y//2
    crop_frame = undst[h+h//5:y, :]
    # cv2.imshow("cropped", crop_frame)
    # cv2.waitKey(0) 
    # cv2.imwrite("OG homo.jpg", crop_frame)
    
    edges = canny2(crop_frame)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)
    # cv2.imwrite("Canny.jpg", edges)

# Get the homograpghy matrix and lane corners     
h_matrix, lane_corners = find_homo(edges, crop_frame)
    

# Create video frame generated frames
def visual(frames):
    path = '/home/elliottmcg/Desktop/School/Second Semester/Perception_673/frames/'
    img=[]
    for i in range(0,frames):
        img.append(cv2.imread(path+f'frame{i}.jpg'))
        
    height,width,layers=img[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if a == 1:
        name = 'data_set1_video_solution.avi'
    if a == 2:
        name = 'challenge_video_solution.avi'
    video=cv2.VideoWriter(name, fourcc, 15,(width,height))

    for j in range(0,len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()
    return

done = 0
count = 0
# cap.set(1, 90)

# loop through the videos
while cap.isOpened():
    # Break loop at end of video
    success, frame = cap.read()
    if success == False:
        break
    
    # Undistort frame
    undst = undistort(frame)
    
    # Apply lane detection to selected video
    if a == 1:
        
        # Crop sky out of frame
        y = int(undst.shape[0])
        h = y//3
        crop_frame = undst[h+h//3:y, :]
        # cv2.imshow("cropped", crop_frame)
        # cv2.waitKey(1) 
        
        # Warp frame
        warp = cv2.warpPerspective(crop_frame,h_matrix,(600,600))
        # cv2.imshow('warped', warp)
        # cv2.waitKey(0)

        # Call all lane detection functions
        warp_lines = warp.copy()
        peaks, tops, line_1_pts, line_2_pts = histogram1(warp)
        lane_image, direction = draw_lines(warp_lines, peaks, tops, line_1_pts, line_2_pts)
        count = return_image(lane_image, lane_corners, crop_frame, direction, count)
        
    if a == 2:
        
        y = int(undst.shape[0])
        h = y//2
        crop_frame = undst[h+h//5:y, :]
        # cv2.imshow("cropped", crop_frame)
        # cv2.waitKey(1) 
        # cv2.imwrite("cropped.jpg", crop_frame)
        
        warp = cv2.warpPerspective(crop_frame,h_matrix,(600,600))
        # cv2.imshow('warped', warp)
        # cv2.waitKey(0)
        # cv2.imwrite("warp.jpg", warp)
        
        warp_lines = warp.copy()
        peaks, tops, line_1_pts, line_2_pts = histogram2(warp)
        lane_image, direction = draw_lines(warp_lines, peaks, tops, line_1_pts, line_2_pts)
        count = return_image(lane_image, lane_corners, crop_frame, direction, count)
    
    # Break loop after one frame
    # if done == 0:
    #     break
    # done += 1
    
###
# Part of video create, leave commented out    
# visual(count)
###

cv2.destroyAllWindows()
    