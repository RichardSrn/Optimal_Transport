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

    print(sum([all([p in file for p in param]) for file in files]))

    for file in files:
        print(all([p in file for p in param]))
        if all(p in file for p in param):
            print(f"{file} already exists.")
            # return True

    return False


dbs_files = get_files("./results/debiased_sink_bary")

params = [(0.00001, int(1e2)),
          (0.00001, int(1e6) ),
          (0.00001, int(1e3)),
          (.001,    int(1e6) ),
          (.005,    int(1e6) ),
          (.01,     int(1e6) ),
          (.01,     int(1e8) ),
          (.01,     int(1e5)),
          (.05,     int(1e7) ),
          (.05,     int(1e5)),
          (.2,      2500),
          (.2,      3000),
          (.2,      4000),
          (.5,      500),
          (.5,      750),
          (.5,      1000),
          (.7,      100),
          (.7,      750),
          (.7,      1000)
         ]
         
for epsilon, max_iter in params:
    if not check_in_files(dbs_files, eps=epsilon, iter=max_iter):
        print("didn't exist yet.")