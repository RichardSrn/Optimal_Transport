#! /usr/bin/env python3

from debiased_sink_bary import debiased_sink_bary
from kbcm_bary import kbcm_bary
from entropic_reg_bary import entropic_reg_bary
from tlp_bary import tlp_bary

import sys

def main():
    print("\n\n\n\nRunning DSB")
    for epsilon in [0.1,0.6] :
        for max_iter in [100,300,500] :
            print(f"\nepsilon = {epsilon} ; max_iter = {max_iter}")
            try :
                debiased_sink_bary(epsilon = epsilon, max_iter = max_iter, intensity = "minmax", plot=False)
                print("Execution ended normally.")
            except :
                ertype = sys.exc_info()[0]
                erdescription = sys.exc_info()[1]  
                print(f"WARNING - DSB - didn't run with parameters epsilon={epsilon}, max_iter={max_iter}")
                print("Error type :",ertype)
                print("Error description :", erdescription)

    print("\n\n\n\n","-"*50)

    print("\n\n\n\nRunning ERB")
    for reg in [0.075,0.4,1,4] :
        print(f"\nregression = {reg}")
        try :
            entropic_reg_bary(reg=reg, sample = 100, plot=False)
            print("Execution ended normally.")
        except :
            ertype = sys.exc_info()[0]
            erdescription = sys.exc_info()[1] 
            print(f"WARNING - ERB - didn't run with parameters reg={reg}")
            print("Error type :",ertype)
            print("Error description :", erdescription)

    print("\n\n\n\n","-"*50)

    print("\n\n\n\nRunning KCBM")
    for reg in [0.75,0.4,1,4] :
        for max_iter in [100,300,500] :
            print(f"\nregression = {reg} ; max_iter = {max_iter}")
            try :
                kbcm_bary(reg = reg, c = -.7, max_iter = 100, save=True, intensity = "maxmin", plot=False)
                print("Execution ended normally.")
            except :
                ertype = sys.exc_info()[0]
                erdescription = sys.exc_info()[1] 
                print(f"WARNING - KCBM - didn't run with parameters reg={reg}, max_iter={max_iter}")
                print("Error type :",ertype)
                print("Error description :", erdescription)

    print("\n\n\n\n","-"*50)

    print("\n\n\n\nRunning TLp")
    for reg in [0.75,0.4,1,4] :
        for eta in [.001, .1, .05, .7] :
            for intensity in ["minmax", "zeroone"] :
                print(f"\nregression = {reg}, eta = {eta}, intensity = {intensity}")
                try :
                    tlp_bary(reg = reg, eta = eta, intensity = intensity, plot=False)#, outItermax = 20, inItermax=100, outstopThr=1e-10, instopThr=1e-10)
                    print("Execution ended normally.")
                except :
                    ertype = sys.exc_info()[0]
                    erdescription = sys.exc_info()[1] 
                    print(f"WARNING - TLp - didn't run with parameters reg={reg}, eta = {eta}, intensity = {intensity}")
                    print("Error type :",ertype)
                    print("Error description :", erdescription)
            
main()
