#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bananasacks
"""

import os
#os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")

import matplotlib.pyplot as plt
import numpy as np
import cv2


data00 = np.load("./results/debiased_sink_bary/bary_noiselvl_0.000_eps_0.7_iter_500_imgs_5_intensity_minmax_noise_lvls_6.npy")
data05 = np.load("./results/debiased_sink_bary/bary_noiselvl_0.050_eps_0.7_iter_500_imgs_5_intensity_minmax_noise_lvls_6.npy")
data10 = np.load("./results/debiased_sink_bary/bary_noiselvl_0.100_eps_0.7_iter_500_imgs_5_intensity_minmax_noise_lvls_6.npy")
data20 = np.load("./results/debiased_sink_bary/bary_noiselvl_0.200_eps_0.7_iter_500_imgs_5_intensity_minmax_noise_lvls_6.npy")
data50 = np.load("./results/debiased_sink_bary/bary_noiselvl_0.500_eps_0.7_iter_500_imgs_5_intensity_minmax_noise_lvls_6.npy")
data_1 = np.load("./results/debiased_sink_bary/bary_noiselvl_1.000_eps_0.7_iter_500_imgs_5_intensity_minmax_noise_lvls_6.npy")

data = [data00, data05, data10, data20, data50, data_1]
nanmin = np.nanmin(data)
nanmax = np.nanmax(data)
#print(nanmin)
#print(nanmax)


#plt.figure(1, figsize=(15, 10))
#k=1
#for d in data:
#    plt.subplot(2, 3, k)
#    plt.imshow(d, vmin=nanmin, vmax=nanmax)
#    k+=1
   
 
#plt.figure(2, figsize=(15, 10))
#fig2 = plt.imshow(data[0])
   
 # Use cv2.minMaxLoc to find the brightest and darkest points in the image
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(data[0])
print(minVal)
print(maxVal)
print(maxLoc)
 # Draw the result in the image
circle = cv2.circle(data[0], maxLoc, 5, (0, 0, 0), 1)
   
plt.figure(3, figsize=(15, 10))
cv2.imshow('fig2', circle)
plt.imshow(circle, vmin=nanmin, vmax=nanmax)

plt.figure(1, figsize=(15, 10))
k=1
for d in data:
    plt.subplot(2, 3, k)
    circle = cv2.circle(d, maxLoc, 5, (0, 0, 0), 1)
    plt.imshow(circle, vmin=nanmin, vmax=nanmax)
    k+=1



img = data[0]
edges = cv2.Canny(img,50,50)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()