#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creating the plot for a sample of the edge detection process
"""
import os
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")

import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
import skimage.feature
import skimage.viewer
import tabulate
import math

#dataset: "./results/debiased_sink_bary/bary_noiselvl_1.000_eps_0.3_iter_100_imgs_200_intensity_minmax.npy"


eps = str(0.3)
iters = str(100)
title = "./results/debiased_sink_bary/bary_noiselvl_"
params = "_eps_"+eps+"_iter_"+iters+"_imgs_200_intensity_minmax"



k = 1 
m = 3
data = []
Center=[]
Dist=[]
Diam=[]
Change=[]
row = []
X = []
Y = []
D = []

noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
plt.figure(1, figsize=(15, 10))
plt.title("bary_lvl_"+params)
for n in noise_lvls: 
    barys = np.load(title + n + params + ".npy") 
    plt.subplot(2, m, k)
    
    ##normalizing and scaling data to work with canny
    barys /= barys.max()
    barys = 255 * barys # Now scale by 255
    img = barys.astype(np.uint8)

    # median filter
    median = cv2.medianBlur(img, 1)
    
    #CANNY but should update the sigmas
    canny = cv2.Canny(img,0,45)
    # transpose canny image to compensate for following numpy points as y,x
    canny_t = cv2.transpose(canny) 
    # get canny points
    # numpy points are (y,x)
    points = np.argwhere(canny_t>0)
    #code using ellipse
    # fit ellipse and get ellipse center, minor and major diameters and angle in degree
    ellipse = cv2.fitEllipse(points)
    (x,y), (d1,d2), angle = ellipse
    print('center: (', x,y, ')', 'diameters: (', d1, d2, ')')
    # draw ellipse
    result = img.copy()
    cv2.ellipse(result, (int(x),int(y)), (int(d1/2),int(d2/2)), angle, 0, 360, (0,0,0), 1)
    # draw circle on copy of input of radius = half average of diameters = (d1+d2)/4
    rad = int((d1+d2)/4)
    xc = int(x)
    yc = int(y)
    print('center: (', xc,yc, ')', 'radius:', rad)
    cv2.circle(result, (xc,yc), rad, (0,255,0), 1)
    
    #if intensity == "zeroone":
    #plt.imshow(barys, vmin=0, vmax=.05)
    #elif intensity == "minmax":
    #plt.imshow(barys, vmin=vmin, vmax=vmax)
    #else:
    row=[]
    X.append(xc)
    Y.append(yc)
    D.append((d1+d2)/2)
    row.append("("+str(xc)+","+str(yc)+")")
    row.append(int(math.sqrt((int(yc)-Y[0])**2+(int(xc)-X[0])**2)))
    row.append(int((d1+d2)/2))
    row.append(int((d1+d2)/2)-int(D[0]))
    data.append(row)
    k += 1
    plt.imshow(result, vmin=0, vmax=255)

plt.show()


print(data)



plt.figure(2, figsize=(10, 10))
plt.axis("off")
plt.xticks([])
plt.yticks([])
plt.box(on=None)
rows = ["Image 1", "Image 2", "Image 3", "Image 4", "Image 5"]
columns = ["Center", "Dist to Original Center", "Diameter", "Change from Original Diameter" ]
plt.table(cellText=data,
    rowLabels=rows,
    colLabels=columns,
    loc='center').scale(1.5, 3)
plt.show()
    

