#! /usr/bin/env python3

import numpy as np
import pandas as pd
from criteria import criteria
import os
import re

def parse_key(key) :
    true_key = { "debiased_sink_bary" : "debiased_sinkhorn",
                    "entropic_reg_bary"  : "entropic_regularized",
                    "kbcm_bary"          : "kbcm",
                    "tlp_bary"           : "Tlp",
                    "lvl"                : "noise level"}
    if key in true_key.keys():
        return true_key[key]
    else :
        return key

def get_files(path = "./data"):
    directories = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    directories.remove('noisefrees_control')

    tree = {d : [] for d in directories}

    for d in directories :
        tree[d] = [f for f in os.listdir(os.path.join(path,d)) if os.path.isfile(os.path.join(path,d,f))]
        tree[d] = [file for file in tree[d] if file[-4:] == ".npy"]
        tree[d].sort()

    return tree

def compare(path = "./data/", save=True, save_path="./results/"):
    tree = get_files(path)
    data = {d : np.array([np.load(os.path.join(path,d,file)) for file in tree[d]]) for d in tree.keys()}
    
    df = pd.DataFrame(columns=[ "noise level",
                                "max amplitude",
                                "above-thld pixels",
                                "above-thld pixels std",
                                "algorithm"])
    
    for d in tree.keys() :
        for file_name, barycenter in zip(tree[d], data[d]) :
            param = dict({"algorithm" : parse_key(d)})
            
            name = file_name[:-4]
            groups = re.findall("([a-zA-Z]+_[+-]?\d+(?:\.\d+)?)",name)

            for g in groups:
                key = re.findall("([a-zA-Z]+)",g)
                value = re.findall("([+-]?\d+(?:\.\d+)?)",g)
                if key[0] != "mean" :
                    param[parse_key(key[0])] = float(value[0])

            max_ampl, pixels, std = criteria(barycenter)
            param["max amplitude"] = max_ampl
            param["above-thld pixels"] = pixels
            param["above-thld pixels std"] = std
            df = df.append(param, ignore_index=True)

    if save :
        df.to_csv(os.path.join(save_path,"DataFrame_summary.csv"))

    return df


                            

            

                


            




if __name__ == "__main__":
    compare("/home/lmint/Documents/programmation/python/Optimal_Transport/deliverable_clean/test_algos/results")

