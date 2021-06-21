#! /usr/bin/env python3

from debiased_sink_bary import debiased_sink_bary
from kbcm_bary import kbcm_bary
from entropic_reg_bary import entropic_reg_bary
from tlp_bary import tlp_bary

def main():
    print("Running DSB")
    for epsilon in [0.1,0.6] :
        for max_iter in [100,300,500] :
            print(f"epsilon = {epsilon} ; max_iter = {max_iter}")
            try :
                debiased_sink_bary(epsilon = epsilon, max_iter = max_iter, intensity = "minmax", plot=False)
                print("No error.")
            except :
                print(f"DSB - didn't run with parameters epsilon={epsilon}, max_iter={max_iter}")
    
    print("Running ERB")
    for reg in [0.075,0.4,1,4] :
        print(f"regression = {reg}")
        try :
            entropic_reg_bary(reg=reg, sample = 100, plot=False)
            print("No error.")
        except :
            print(f"ERB - didn't run with parameters reg={reg}")

    print("Running KCBM")
    for reg in [0.75,0.4,1,4] :
        for max_iter in [100,300,500] :
            print(f"regression = {reg} ; max_iter = {max_iter}")
            try :
                kbcm_bary(reg = reg, c = -.7, max_iter = 100, save=True, intensity = "maxmin", plot=False)
                print("No error.")
            except :
                print(f"KCBM - didn't run with parameters reg={reg}, max_iter={max_iter}")
    
    print("Running TLp")
    for reg in [0.75,0.4,1,4] :
        print(f"regression = {reg}")
        try :
            tlp_bary(reg = .05, eta = .1, intensity ="minmax", plot=False)#, outItermax = 20, inItermax=100, outstopThr=1e-10, instopThr=1e-10)
            print("No error.")
        except :
            print(f"TLp - didn't run with parameters reg={reg}")


main()