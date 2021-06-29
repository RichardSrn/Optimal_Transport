# -*- coding: utf-8 -*-
"""
See if this method removes high frequency noise
https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
"""
import os
os.chdir("/Users/bananasacks/Desktop/Optimal Transport Internship/Optimal_Transport/deliverable_clean/test_algos")

import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
import skimage.feature
import skimage.viewer

#KBCM
#kbcm reg = [.75, 1, 4] is empty (nan for all)
#kbcm reg = [.4] gives ok results
#reg = str(.4)
#c = str(-0.7)
#title = "./results/kbcm_bary/bary_noiselvl_"
#params = "_reg_"+reg+"_c_"+c+"_iters_100_imgs_5_intensity_maxmin_noise_lvls_6"


#TLP
#tlp reg = [.4]  eta = [.001, 05, .1, .7] gives all very similar results (thinking eta does not make a huge difference here)
#tlp reg = [.05] eta = [.1] give good results for lower noise levels (would be worth it to do more around this reg level or below)
#tlp reg = [.75] eta = [.001, .05, .1, .7]  gives all very similar results
#tlp reg = [1] eta = [.001, .05, .1, .7] gives all very similar results to .75 and to .4 (with .4 being slightly better)
#tlp reg = [4] eta = [.001, .05, .1, .7] gives all very similar results
#reg = str(4)
#eta = str(.001)
#title = "./results/tlp_bary/bary_noiselvl_"
#params = "_reg_"+reg+"_eta_"+eta+"_outer-inner_10-100_imgs_5_intensity_minmax_noise_lvls_6"

#Debiased
#sink all 0 noise levels are empty (nan for  noise = 0.000)
#sink reg = [.1]  iters = [100, 300, 500] all have NAN for 0 noise, gets slightly better with more iterations
#sink reg = [.6]  iters = [100, 300, 500] all have NAN for 0 noise, gets slightly better with more iterations
#interesting with low epsilon (0.1 and below) at high iterations (1e6 and above)
eps = str(0.6)
iters = str(500)
title = "./results/debiased_sink_bary/bary_noiselvl_"
params = "_eps_"+eps+"_iter_"+iters+"_imgs_5_intensity_minmax_noise_lvls_6"



k = 1 
m = 3
noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
plt.figure(1, figsize=(15, 10))
plt.title("bary_lvl_"+params)
for n in noise_lvls: 
    barys = np.load(title + n + params + ".npy")
    print(n)
    print(barys) 
    plt.subplot(2, m, k)
    ##added vmin and vmax so all plots have same itensity scale
    k += 1
    #if intensity == "zeroone":
    #plt.imshow(barys, vmin=0, vmax=.05)
    #elif intensity == "minmax":
    #plt.imshow(barys, vmin=vmin, vmax=vmax)
    #else:
    plt.imshow(barys)
plt.show()



"""
#img = cv2.imread("./results/kbcm_bary/plots_kbcm_bary/kbcm_bary_noiselvl_1.000_reg_0.25_c_-0.7_iters_100_imgs_5_intensity_maxmin_noise_lvls_6")

plt.figure(1, figsize=(15, 10))
plt.imshow(data)


print(data)
print(data.max())

data /= data.max()
data = 255 * data # Now scale by 255
img = data.astype(np.uint8)
print(data.max())
print(img.max())
plt.imshow(img)

# median filter
#img = cv2.medianBlur(data, 3)



#img = cv2.medianBlur(data, 3)




#img = skimage.io.imread(fname="./results/kbcm_bary/bary_noiselvl_0.200_reg_0.25_c_-0.7_iters_100_imgs_5_intensity_maxmin_noise_lvls_6.npy", as_gray=True)
#img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img,0,.1)


plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
"""
