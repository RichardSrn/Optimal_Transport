#! /usr/bin/env python3

from debiased_sink_bary import debiased_sink_bary
from kbcm_bary import kbcm_bary
from entropic_reg_bary import entropic_reg_bary
from tlp_bary import tlp_bary
from multiprocessing import Process, Queue
import sys
from os import listdir
from os.path import join,isfile


# def debiased_sink_bary(**kwars):
#     print("debiased_sink_bary")
#     for k,v in kwars.items():
#         print(k,":",v)
# def kbcm_bary(**kwars):
#     print("kbcm_bary")
#     for k,v in kwars.items():
#         print(k,":",v)
# def entropic_reg_bary(**kwars):
#     print("entropic_reg_bary")
#     for k,v in kwars.items():
#         print(k,":",v)
# def tlp_bary(**kwars):
#     print("tlp_bary")
#     for k,v in kwars.items():
#         print(k,":",v)


Q = Queue()

def get_files(path = "./results"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]       
    onlyfiles.sort()
    
    return onlyfiles

def check_in_files(files, **parameters) :
    param = []
    for k,v in parameters.items():
        param.append(str(k)+'_'+str(v))

    print(param)

    for file in files :
        if all( p in file for p in param ) :
            print(f"{file} already exists.")
            return True

    return False



def run_dsb(epsilon, max_iter) :
    logs=[]
    logs.append(f"\nepsilon = {epsilon} ; max_iter = {max_iter}"+"\n")
    try :
        debiased_sink_bary(epsilon = epsilon, max_iter = max_iter, intensity = "minmax", plot=False)
        logs.append("Execution ended normally."+"\n")
    except :
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1]  
        logs.append(f"WARNING - DSB - didn't run with parameters epsilon={epsilon}, max_iter={max_iter}"+"\n")
        logs.append(f"Error type : {ertype}"+"\n")
        logs.append(f"Error description : {erdescription}"+"\n")
    Q.put(' '.join(logs))



def run_entropic(reg) :
    logs=[]
    logs.append(f"\nregression = {reg}"+"\n")
    try :
        entropic_reg_bary(reg=reg, sample = 100, plot=False)
        logs.append("Execution ended normally."+"\n")
    except :
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1] 
        logs.append(f"WARNING - ERB - didn't run with parameters reg={reg}"+"\n")
        logs.append(f"Error type : {ertype}"+"\n")
        logs.append(f"Error description : {erdescription}"+"\n")
    Q.put(' '.join(logs))



def run_kbcm(reg, max_iter, c, intensity) :
    logs=[]
    logs.append(f"\nregression = {reg} ; max_iter = {max_iter} ; c = {c} ; intensity = {intensity}"+"\n")
    try :
        kbcm_bary(reg = reg, c = c, max_iter = 100, save=True, intensity = "maxmin", plot=False)
        logs.append("Execution ended normally."+"\n")
    except :
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1] 
        logs.append(f"WARNING - KBCM - didn't run with parameters reg={reg}, max_iter={max_iter}"+"\n")
        logs.append(f"Error type : {ertype}"+"\n")
        logs.append(f"Error description : {erdescription}"+"\n")
    Q.put(' '.join(logs))


def run_tlp(reg, eta, intensity):
    logs=[]
    logs.append(f"\nregression = {reg}, eta = {eta}, intensity = {intensity}"+"\n")
    try :
        tlp_bary(reg = reg, eta = eta, intensity = intensity, plot=False)#, outItermax = 20, inItermax=100, outstopThr=1e-10, instopThr=1e-10)
        logs.append("Execution ended normally."+"\n")
    except :
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1] 
        logs.append(f"WARNING - TLp - didn't run with parameters reg={reg}, eta = {eta}, intensity = {intensity}"+"\n")
        logs.append(f"Error type : {ertype}"+"\n")
        logs.append(f"Error description : {erdescription}"+"\n")
    Q.put(' '.join(logs))



def main(algo = None):
    dbs_files = get_files("./results/debiased_sink_bary")
    entr_files = get_files("./results/entropic_reg_bary")
    kbcm_files = get_files("./results/kbcm_bary")
    tlp_files = get_files("./results/tlp_bary")

    algo_to_run = [0 for i in range(4)]

    if algo == None :
        algo_to_run = [1 for i in range(4)]
    else :
        conv = {"dsb"    : 0,
                "0"      : 0,
                "ent"    : 1,
                "1"      : 1,
                "kbcm"   : 2,
                "2"      : 2,
                "tlp"    : 3,
                "3"      : 3}
        algo_to_run[conv[algo]] = 1


    if algo_to_run[0] :
        ############
        # debiased #
        ############
        processes = []
        Q.put("\n\n\n\nRunning DSB"+"\n")

        params = [(.001,    int(1e6) ),
                  (.005,    int(1e6) ),
                  (.01,     int(1e6) ),
                  (.01,     int(1e8) ),
                  (.05,     int(1e7) ),
                  (.2,      2500     ),
                  (.2,      3000     ),
                  (.2,      4000     ),
                  (.5,      500      ),
                  (.5,      750      ),
                  (.5,      1000     ),
                  (.7,      100      ),
                  (.7,      750      ),
                  (.7,      1000     )]

        for epsilon,max_iter in params :
            if not check_in_files(dbs_files,eps = epsilon, iter = max_iter) :
                processes.append( Process(target=run_dsb, args=(epsilon, max_iter,)) )
                processes[-1].start()

        for proc in processes :
            proc.join()
        Q.put("\n\n\n\n"+"-"*50+"\n")

        with open("./logs0.txt", "w") as file :
            while not Q.empty() :
                file.write(Q.get())


    if algo_to_run[1] :
        ############
        # entropic #
        ############
        processes = []
        Q.put("\n\n\n\nRunning ERB"+"\n")
        for reg in [0.075, 0.4, 1, 4] :
            for metric in ["sqeuclidean", "cityblock"] :
                if not check_in_files(entr_files, reg = reg) :
                    processes.append( Process(target=run_entropic, args=(reg,metric)) )
                    processes[-1].start()

        for proc in processes :
            proc.join()
        Q.put("\n\n\n\n"+"-"*50+"\n")


        with open("./logs1.txt", "a") as file :
            while not Q.empty() :
                file.write(Q.get())


    if algo_to_run[2] :
        ########
        # kbcm #
        ########
        processes = []
        Q.put("\n\n\n\nRunning KBCM"+"\n")
        for reg in [.001, .01, .05, .1, .25, .4, .5, .6, .9] :
            for max_iter in [100,500] :
                for c in [-.5] :
                    for intensity in ["minmax"] :
                        if not check_in_files(kbcm_files, reg = reg, iters = max_iter) :
                            processes.append( Process(target=run_kbcm, args=(reg, max_iter,c,intensity)) )
                            processes[-1].start()
            
        for proc in processes :
            proc.join()
        Q.put("\n\n\n\n"+"-"*50+"\n")

        with open("./logs2.txt", "a") as file :
            while not Q.empty() :
                file.write(Q.get())


    if algo_to_run[3] :
        #######
        # TLp #
        #######
        processes = []
        Q.put("\n\n\n\nRunning TLp"+"\n")
        for reg in [.001, .01, .05, .1, .5, .9] :
            for eta in  [.001, .1, .05, .7] :
                for intensity in ["minmax", "zeroone"] :
                    if not check_in_files(tlp_files, reg = reg, eta = eta, intensity = intensity) :
                        processes.append( Process(target=run_tlp, args=(reg, eta, intensity,)) )
                        processes[-1].start()
            
        for proc in processes :
            proc.join()

        with open("./logs3.txt", "a") as file :
            while not Q.empty() :
                file.write(Q.get())





if __name__=="__main__"   :
    if len(sys.argv) == 1:
        main()
    else :
        main(algo = sys.argv[1])
