#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge detection works with no-noise and with high levels of noise

ToDo:
    *Learn how to make subplots better, using ax and fig
    xget the center and radius to print in the 4th subplot
    *make the comparison to the no-noise in the 4th subplot
    *make a plot of 5 with the radius on the original plot and have the 6th be a table of info
    *automate the high and low paramaters based on sigma
    *test different values of the kernel
    *make chart of 5, have original bcenter and actual bcenter on each plot with a table showing the differences a
        at the end
    *work on removing noise, gaussian smoothing etc so that the original noisy images work well
    *might need to do the smoothing in case the brightest spot isn't in the barycenter?
    *run it on real output data
    *
"""
#https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
#https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
#https://stackoverflow.com/questions/66571431/python-cv2-determine-radius-of-bright-spot-in-image
#https://docs.opencv.org/4.5.2/d4/d13/tutorial_py_filtering.html

#hugh circles https://dsp.stackexchange.com/questions/5930/detection-of-a-circle-in-noisy-image-data

import os
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/pascalle_s_drafts/test_algos_draft")

import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
import skimage.feature
import skimage.viewer


"""these two examples break the system
#not a perfect circle
#data = np.load("./results/debiased_sink_bary/bary_noiselvl_1.000_eps_0.7_iter_500_imgs_5_intensity_minmax_noise_lvls_6.npy")

#not a lot of definition
data = np.load("./results/debiased_sink_bary/bary_noiselvl_0.200_eps_0.5_iter_1000_imgs_5_intensity_minmax_noise_lvls_6.npy")
"""
#noisydata example (does not work with the just plain noisy data)
#data = np.load("./data/artificial_data_noiselvl_0.100.npy")
#data = data[10]

"""Can try using a threshold
https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
Or simple blob detector
https://learnopencv.com/blob-detection-using-opencv-python-c/
"""

##Good example to use for final one, it's KBCM
#kbcm_bary_noiselvl_1.000_reg_0.1_c_-0.25_iters_10_imgs_5_intensity_maxmin_noise_lvls_6.png

#kbcm example
#data = np.load("./results/kbcm_bary/bary_noiselvl_0.100_reg_0.1_c_-0.25_iters_10_imgs_5_intensity_maxmin_noise_lvls_6.npy")

#data = np.load("./results/kbcm_bary/bary_noiselvl_0.100_reg_0.075_c_-0.7_iters_10_imgs_5_intensity_maxmin_noise_lvls_6.npy")

#no noise
#data = np.load("./results/debiased_sink_bary/bary_noiselvl_0.000_eps_0.3_iter_100_imgs_200_intensity_minmax.npy")

#adding noise to check results
#data = np.load("./results/debiased_sink_bary/bary_noiselvl_0.100_eps_0.3_iter_100_imgs_200_intensity_minmax.npy")

#adding noise to check results
#data = np.load("./results/debiased_sink_bary/bary_noiselvl_0.500_eps_0.3_iter_100_imgs_200_intensity_minmax.npy")

#adding noise to check results
data = np.load("./results/debiased_sink_bary/bary_noiselvl_1.000_eps_0.3_iter_100_imgs_200_intensity_minmax.npy")

#img = cv2.imread("./results/kbcm_bary/plots_kbcm_bary/kbcm_bary_noiselvl_1.000_reg_0.25_c_-0.7_iters_100_imgs_5_intensity_maxmin_noise_lvls_6")

plt.figure(1, figsize=(15, 10))
#plt.imshow(data)

##normalizing and scaling data to work with canny
#print(data)
#print(data.max())

data /= data.max()
data = 255 * data # Now scale by 255
img = data.astype(np.uint8)
#print(data.max())
#print(img.max())
#plt.imshow(img)


#smoothing the image to see if there are better results with noise
#thus kernel size must be odd and greater than 1 for simplicity.
# median filter
median = cv2.medianBlur(img, 1)

#using a gaussian filter (kernel is positive and odd also)
#median = cv2.GaussianBlur(img,(1,1),0)


#there's code to automate this based on sigma
#https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
canny = cv2.Canny(img,0,20)
#canny = skimage.feature.canny(img, sigma=3)
#canny_t = canny


plt.subplot(121),plt.imshow(median)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# transpose canny image to compensate for following numpy points as y,x
canny_t = cv2.transpose(canny)

# get canny points
# numpy points are (y,x)
points = np.argwhere(canny_t>0)

"""code using circle
# get min enclosing circle
center, radius = cv2.minEnclosingCircle(points)
print('center:', center, 'radius:', radius)

# draw circle on copy of input
result = img.copy()
x = int(center[1])
y = int(center[0])
rad = int(radius)
cv2.circle(result, (x,y), rad, (0,255,0), 1)
"""





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

# write results
#cv2.imwrite("canny_ellipse.jpg", canny)
#cv2.imwrite("ellipse_circle.jpg", result)



# show results
plt.subplot(2, 2, 1)
plt.imshow(data, vmin=0,vmax=255)
plt.title("original")
plt.subplot(2, 2, 2)
plt.imshow(canny)
plt.title("canny")
plt.subplot(2, 2, 3)
plt.imshow(result)
plt.title("results")
#cv2.waitKey(0)
plt.subplot(2, 2, 4)
plt.axis("off")
plt.xticks([])
plt.yticks([])
plt.box(on=None)
data=[[x,int(d1)],
      [y,int(d2)]]
column_labels=["Center", "radius"]
plt.axis('tight')
plt.axis('off')
plt.table(cellText=data,colLabels=column_labels, rowLabels=["Original","Distance"], loc='center').scale(1.5, 1.5)
plt.show()








