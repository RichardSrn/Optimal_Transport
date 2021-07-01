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
import torch

#KBCM
#kbcm reg = [.75, 1, 4] is empty (nan for all)
#kbcm reg = [.4] gives ok results
#reg = str(.4)
#c = str(-0.7)
#title = "./results/kbcm_bary/bary_noiselvl_"
#params = "_reg_"+reg+"_c_"+c+"_iters_100_intensity_maxmin"


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
#eps = str(.1)#[.1]
#iters = str(100)#[100,300,500]

#for e in eps:
#    for i in iters:
        
#title = "./results/debiased_sink_bary/bary_noiselvl_"
#params = "_eps_"+str(eps)+"_iter_"+str(iters)+"_intensity_minmax"

#bary_noiselvl_0.000_eps_0.1_iter_100_intensity_minmax.npy

def quickplot(algo, reg, second_param):   
    for r in reg:
        for p in second_param: 
            j=1
            k = 1 
            vmin = []
            vmax = [] 
            if algo == "kbcm":
                title = "./results/kbcm_bary/bary_noiselvl_"
                params = "_reg_"+str(r)+"_c_"+str(p)+"_iters_100_intensity_maxmin"
            elif algo == "tlp":
                title = "./results/tlp_bary/bary_noiselvl_"
                params = "_reg_"+str(r)+"_eta_"+str(p)+"_outer-inner_10-100_intensity_minmax"
            elif algo == "sink":
                title = "./results/debiased_sink_bary/bary_noiselvl_"
                params = "_eps_"+str(r)+"_iter_"+str(p)+"_intensity_minmax"
            else:
                print("something went wrong")
                
            path = title + "0.000" + params + ".npy"
            if os.path.isfile(path):
                
                noise_lvls = ["0.000", "0.100", "0.200", "0.500", "1.000"]
                plt.figure(j, figsize=(15, 10))
                #fig, ax = plt.subplots(figsize=(8,8))
                #plt.suptitle(title[10:-15] + params)
                for n in noise_lvls: 
                    barys = np.load(title + n + params + ".npy")
                    #print(n)
                    #print(barys) 
                    nanmin = np.nanmin(barys)
                    nanmax = np.nanmax(barys)
                    print(nanmin)
                    print(nanmax)
                #Finding max and min intensities for consistent plotting
                    #Finding the Min intensity with NAN handler
                    #print(vmin)
                    #print(nanmin)

                    if vmin == []:
                        if np.isnan(nanmin) == True:
                            vmin = []
                        else:
                            vmin = nanmin
                    ##If NAN, do nothing
                    ##If min(bary) > vmin, do nothing
                        #print(vmin)
                    else:
                        if np.isnan(nanmin) == False:
                            barymin = nanmin
                            if  barymin < vmin:
                                vmin = barymin 
                        #print(vmin)
                    if np.isnan(vmin):
                        vmin = 0
                    #print(vmin)
                    #Finding the Max intensity with NAN handler
                    if vmax == []:
                        if np.isnan(nanmax) == True:
                            vmax = []
                        else:
                            vmax = nanmax
                    ##If NAN, do nothing
                    ##If max(bary) < vmax, do nothing
                    else:
                        if np.isnan(nanmax) == False:
                            barymax = nanmax
                            if  barymax > vmax:
                                vmax = barymax     
                    if np.isnan(vmax):
                        vmax = 1
     
                    plt.subplot(2, 3, k)
                    k += 1
                    plt.imshow(barys)#, vmin=vmin, vmax=vmax)
                    
                plt.suptitle(title[10:-15] + params + "     max_amplitude: "+ str(round(vmax,5)))
                plt.savefig("results/report_plots/"+title[10:-15] + params + ".png")
                plt.show()
                j+=1
                
"""
if __name__ == "__main__":
    algo = "kbcm"
    reg = [0.1, 0.01, 0.001, 0.4, 0.5, 0.05, 0.6, 0.9, 0.25, 0.75, 1, 4]
    c = [-0.5, -0.7]
    quickplot(algo, reg, c)
"""    
"""    
if __name__ == "__main__":
    algo = "tlp"
    reg = [0.4, 0.05, 0.75, 1, 4]
    eta = [0.1, 0.001, 0.05, 0.7]
    quickplot(algo, reg, eta)   

      
if __name__ == "__main__":
    algo = "sink"
    reg = [0.1]#, 0.01, 0.001, 0.2, 0.5, 0.005, 0.6, 0.7]
    iters = [100]#,300,500,1000000, 2500, 3000, 4000, 1000]
    quickplot(algo, reg, iters)        
"""
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
