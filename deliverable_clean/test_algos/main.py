#! /usr/bin/env python3

import sys
from multiprocessing import Process, Queue
from os import listdir
from os.path import join, isfile
from time import time

from debiased_sink_bary import debiased_sink_bary
from entropic_reg_bary import entropic_reg_bary
from entropic_reg_bary_convol import entropic_reg_bary_convol
from kbcm_bary import kbcm_bary
from tlp_bary import tlp_bary

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
# def entropic_reg_bary_convol(**kwars):
#     print("entropic_reg_bary_convol")
#     for k,v in kwars.items():
#         print(k,":",v)
# def tlp_bary(**kwars):
#     print("tlp_bary")
#     for k,v in kwars.items():
#         print(k,":",v)


Q = Queue()
T = Queue()


def get_files(path="./results"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]
    onlyfiles.sort()

    return onlyfiles


def check_in_files(files, **parameters):
    param = []
    for k, v in parameters.items():
        param.append(str(k) + '_' + str(v))

    print(param)

    if sum([all([p in file for p in param]) for file in files]) >= 4:
        print(f"{param} already computed for each noise level.")
        return True

    return False


def run_dsb(epsilon, max_iter):
    t = time()
    logs = []
    logs.append(f"\nepsilon = {epsilon} ; max_iter = {max_iter}" + "\n")
    try:
        debiased_sink_bary(epsilon=epsilon, max_iter=max_iter, intensity="minmax", plot=False)
        logs.append("Execution ended normally." + "\n")
    except:
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1]
        logs.append(
            f"WARNING - DSB - didn't run properly with parameters epsilon={epsilon}, max_iter={max_iter}" + "\n")
        logs.append(f"Error type : {ertype}" + "\n")
        logs.append(f"Error description : {erdescription}" + "\n")
    T.put(';'.join([str(epsilon), str(max_iter), str(time() - t)]))
    Q.put(' '.join(logs))


def run_entropic(reg, metric):
    t = time()
    logs = []
    logs.append(f"\nregularization = {reg} ; metric = {metric}" + "\n")
    try:
        entropic_reg_bary(reg=reg, metric=metric, plot=False)
        logs.append("Execution ended normally." + "\n")
    except:
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1]
        logs.append(f"WARNING - ERB - didn't run properly with parameters reg={reg}, metric={metric}" + "\n")
        logs.append(f"Error type : {ertype}" + "\n")
        logs.append(f"Error description : {erdescription}" + "\n")
    T.put(';'.join([str(reg), str(metric), str(time() - t)]))
    Q.put(' '.join(logs))


def run_kbcm(reg, max_iter, c, intensity, samples):
    t = time()
    logs = []
    logs.append(f"\nregularization = {reg} ; max_iter = {max_iter} ; c = {c} ; intensity = {intensity} ; samples = {samples}" + "\n")
    try:
        kbcm_bary(reg=reg, c=c, max_iter=100, save=True, intensity="maxmin", plot=False, samples=samples)
        logs.append("Execution ended normally." + "\n")
    except:
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1]
        logs.append(f"WARNING - KBCM - didn't run properly with parameters reg={reg}, max_iter={max_iter}, samples={samples}" + "\n")
        logs.append(f"Error type : {ertype}" + "\n")
        logs.append(f"Error description : {erdescription}" + "\n")
    T.put(';'.join([str(reg), str(max_iter), str(c), str(intensity), str(samples), str(time() - t)]))
    Q.put(' '.join(logs))


def run_tlp(reg, eta, intensity):
    t = time()
    logs = []
    logs.append(f"\nregularization = {reg}, eta = {eta}, intensity = {intensity}" + "\n")
    try:
        tlp_bary(reg=reg, eta=eta, intensity=intensity,
                 plot=False)  # , outItermax = 20, inItermax=100, outstopThr=1e-10, instopThr=1e-10)
        logs.append("Execution ended normally." + "\n")
    except:
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1]
        logs.append(
            f"WARNING - TLp - didn't run properly with parameters reg={reg}, eta = {eta}, intensity = {intensity}" + "\n")
        logs.append(f"Error type : {ertype}" + "\n")
        logs.append(f"Error description : {erdescription}" + "\n")
    T.put(';'.join([str(reg), str(eta), str(intensity), str(time() - t)]))
    Q.put(' '.join(logs))


def run_entropic_convol(reg):
    t = time()
    logs = []
    logs.append(f"\nregularization = {reg}" + "\n")
    try:
        entropic_reg_bary_convol(reg=reg, plot=False)
        logs.append("Execution ended normally." + "\n")
    except:
        ertype = sys.exc_info()[0]
        erdescription = sys.exc_info()[1]
        logs.append(f"WARNING - ERBc - didn't run properly with parameters reg={reg}" + "\n")
        logs.append(f"Error type : {ertype}" + "\n")
        logs.append(f"Error description : {erdescription}" + "\n")
    T.put(';'.join([str(reg), str(time() - t)]))
    Q.put(' '.join(logs))


def main(algo=None):
    dbs_files = get_files("./results/debiased_sink_bary")
    entr_files = get_files("results/entropic_reg_bary_convol")
    kbcm_files = get_files("./results/kbcm_bary")
    tlp_files = get_files("./results/tlp_bary")

    algo_to_run = [0 for i in range(5)]

    if algo == None:
        algo_to_run = [1 for i in range(5)]
    else:
        conv = {"dsb": 0,
                "0": 0,
                "ent": 1,
                "1": 1,
                "kbcm": 2,
                "2": 2,
                "tlp": 3,
                "3": 3,
                "entc": 4,
                "4": 4}
        algo_to_run[conv[algo]] = 1


    if algo_to_run[0]:
        ############
        # debiased #
        ############
        processes = []
        Q.put("\n\n\n\nRunning DSB" + "\n")

        params = [
                  (.00001,  100),
                  (.00001,  10000),
                  (.00001,  1000000),
                  (.00001,  100000000),
#
                  (.0001,   100),
                  (.0001,   10000),
                  (.0001,   1000000),
                  (.0001,   100000000),
#
                  (.001,    100),
                  (.001,    10000),
                  (.001,    1000000),
                  (.001,    100000000),
#
                  (.005,    100),
                  (.005,    10000),
                  (.005,    1000000),
                  (.005,    100000000),
#
                  (.01,     100),
                  (.01,     10000),
                  (.01,     1000000),
                  (.01,     100000000),
#
                  (.05,     100),
                  (.05,     10000),
                  (.05,     1000000),
                  (.05,     100000000),
#
                  (.2,      100),
                  (.2,      500),
                  (.2,      1000),
                  (.2,      5000),
                  (.2,      10000),
                  (.2,      100000),
                  (.2,      1000000),
#
                  (.5,      100),
                  (.5,      500),
                  (.5,      1000),
                  (.5,      5000),
                  (.5,      10000),
                  (.5,      100000),
                  (.5,      1000000),
#
                  (.7,      100),
                  (.7,      500),
                  (.7,      1000),
                  (.7,      5000),
                  (.7,      10000),
                  (.7,      100000),
                  (.7,      1000000),
#
                  (1,      100),
                  (1,      500),
                  (1,      1000),
                  (1,      5000),
                  (1,      10000),
                  (1,      100000),
                  (1,      1000000)
                 ]
        for epsilon, max_iter in params:
            if not check_in_files(dbs_files, eps=epsilon, iter=max_iter):
                processes.append(Process(target=run_dsb, args=(epsilon, max_iter)))
                processes[-1].start()

        for proc in processes:
            proc.join()
        Q.put("\n\n\n\n" + "-" * 50 + "\n")

        with open("./logs_DBS.txt", "w") as file:
            while not Q.empty():
                file.write(Q.get())

        with open("./times_DBS.csv", "w") as file:
            file.write("epsilon;max_iteration;time")
            while not T.empty():
                file.write(T.get())

    if algo_to_run[1]:
        ############
        # entropic #
        ############
        processes = []
        Q.put("\n\n\n\nRunning ERB" + "\n")
        for reg in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 4]:
            for metric in ["sqeuclidean", "cityblock"]:
                if not check_in_files(entr_files, reg=reg, metric=metric):
                    processes.append(Process(target=run_entropic, args=(reg, metric,)))
                    processes[-1].start()

        for proc in processes:
            proc.join()
        Q.put("\n\n\n\n" + "-" * 50 + "\n")

        with open("./logs_ENT.txt", "w") as file:
            while not Q.empty():
                file.write(Q.get())

        with open("./times_ENT.csv", "w") as file:
            file.write("regularization;metric;time")
            while not T.empty():
                file.write(T.get())

    if algo_to_run[2]:
        ########
        # kbcm #
        ########
        processes = []
        Q.put("\n\n\n\nRunning KBCM" + "\n")
        for reg in [.001, .01, .05, .1, .25, .4, .5, .6, .9]:
            for max_iter in [100, 500]:
                for c in [-.5]:
                    for intensity in ["minmax"]:
                        for samples in [200] :
                            if not check_in_files(kbcm_files, reg=reg, iters=max_iter, samples=samples):
                                processes.append(Process(target=run_kbcm, args=(reg, max_iter, c, intensity, samples)))
                                processes[-1].start()

        for proc in processes:
            proc.join()
        Q.put("\n\n\n\n" + "-" * 50 + "\n")

        with open("./logs_KBCM.txt", "w") as file:
            while not Q.empty():
                file.write(Q.get())

        with open("./times_KBCM.csv", "w") as file:
            file.write("regularization;max_iteration;c;intensity;samples;time")
            while not T.empty():
                file.write(T.get())


    if algo_to_run[3]:
        #######
        # TLp #
        #######
        processes = []
        Q.put("\n\n\n\nRunning TLp" + "\n")
        for reg in [.001, .01, .05, .1, .5, .9]:
            for eta in [.001, .1, .05, .7]:
                for intensity in ["minmax"]:
                    if not check_in_files(tlp_files, reg=reg, eta=eta, intensity=intensity):
                        processes.append(Process(target=run_tlp, args=(reg, eta, intensity,)))
                        processes[-1].start()

        for proc in processes:
            proc.join()

        with open("./logs_TLp.txt", "w") as file:
            while not Q.empty():
                file.write(Q.get())

        with open("./times_TLp.csv", "w") as file:
            file.write("regularization;eta;intensity;time")
            while not T.empty():
                file.write(T.get())

    if algo_to_run[4]:
        ##########################
        # entropic convolutional #
        ##########################
        processes = []
        Q.put("\n\n\n\nRunning ERBc" + "\n")
        for reg in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 4]:
            if not check_in_files(entr_files, reg=reg):
                processes.append(Process(target=run_entropic_convol, args=(reg,)))
                processes[-1].start()

        for proc in processes:
            proc.join()
        Q.put("\n\n\n\n" + "-" * 50 + "\n")

        with open("./logs_ENTc.txt", "w") as file:
            while not Q.empty():
                file.write(Q.get())

        with open("./times_ENTc.csv", "w") as file:
            file.write("regularization;time")
            while not T.empty():
                file.write(T.get())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        main(algo=sys.argv[1])
