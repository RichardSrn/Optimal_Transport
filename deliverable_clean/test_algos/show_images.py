#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def main(stim_id_str="01") :
    noise = ["0.000","0.100","0.200","0.500","1.000"]

    plt.figure(figsize=(15,10))
    i=1
    for n in noise :
        data = np.load("./data/artificial_data_noiselvl_"+n+".npy")[0]
        plt.subplot(2,3,i)
        plt.title("noise level = "+n)
        plt.imshow(data)
        plt.xlabel("")
        plt.xticks()
        plt.ylabel("")
        plt.yticks()
        plt.axis("off")
        i+=1
    data = np.load(f"../../tva_localizer_fmri_data/sample_VO{stim_id_str}.rh.100subjects_data.npy")[0]
    plt.subplot(2,3,i)
    plt.title("sample_VO1.rh.100subjects_data")
    plt.imshow(data)
    plt.xlabel("")
    plt.xticks()
    plt.ylabel("")
    plt.yticks()
    plt.axis("off")


    plt.suptitle("Same image of artificial dataset with different noise level")
    plt.show()

if __name__=='__main__':
    main(12)