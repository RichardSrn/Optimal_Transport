#! /usr/bin/env python3

import numpy as np
import pandas as pd
from criteria import criteria
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import standardize
from adjustText import adjust_text
from math import log10, floor

def get_label(df) :
    def round_sig(x, sig=2):
        return round(x, sig-int(floor(log10(abs(x))))-1)

    df.dropna(axis=1,how="all",inplace=True)

    shorten = {"intensity minmax noise lvls" : "imnl",
               "ababove-thld_pixels"         : "pixels",
               "above-thld_pixels_std"       : "std"}

    text = []
    for c in df.columns :
        values = [str(v) for v in df[c]]

        #try to reduce values' length
        for i in range(len(values)) :
            try :
                values[i] = float(values[i])
                if values[i] == int(values[i]) :
                    values[i] = int(float(values[i]))
                else :
                    values[i] = round_sig(values[i],4)
                    if values[i] < 0.1 :
                        values[i] = "{:e}".format(values[i])
            except :
                pass
            finally :
                values[i] = str(values[i])

        if c in shorten.keys():
            text.append(shorten[c]+'='+", ".join(set(values))+"")
        else :
            text.append(c+'='+", ".join(set(values))+"")
    text ="\n".join( text )
    return text

def parse_key(key) :
    true_key = {    "debiased_sink_bary" : "debiased_sinkhorn",
                    "entropic_reg_bary"  : "entropic_regularized",
                    "kbcm_bary"          : "kbcm",
                    "tlp_bary"           : "Tlp",
                    "imgs"               : "samples",
                    "eps"                : "epsilon",
                    "bary noiselvl"      : "noise_level",
                    "bary lvl"           : "noise_level",
                    "noiselvl"           : "noise_level",
                    "lvl"                : "noise_level"}
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

def collect(path = "./data/", save=True, save_path="./results/"):
    tree = get_files(path)
    data = {d : np.array([np.load(os.path.join(path,d,file)) for file in tree[d]]) for d in tree.keys()}
    
    data = standardize.main(data)

    df = pd.DataFrame(columns=[ "noise_level",
                                "max_amplitude",
                                "barycenter_location",
                                "above-thld_pixels",
                                "above-thld_pixels_std",
                                "algorithm"])
    
    for d in tree.keys() :
        for file_name, barycenter in zip(tree[d], data[d]) :
            param = dict({"algorithm" : parse_key(d)})
            
            name = file_name[:-4]
            groups = re.findall(r"([a-zA-Z](?:[a-zA-Z]|_)+_[+-]?\d+(?:\.\d+)?)",name)

            for g in groups:
                key = ' '.join(re.findall(r"([a-zA-Z]+)",g))
                value = re.findall(r"([+-]?\d+(?:\.\d+)?)",g)
                if key not in ["mean"] :
                    param[parse_key(key)] = float(value[0])

            max_ampl, max_ampl_loc, pixels, std = criteria(barycenter)
            param["max_amplitude"] = max_ampl
            param["barycenter_location"] = max_ampl_loc
            param["above-thld_pixels"] = pixels
            param["above-thld_pixels_std"] = std
            df = df.append(param, ignore_index=True)

    if save :
        df.to_csv(os.path.join(save_path, "DataFrame_summary.csv"), index=False)

    return df

def compare_max_amplitude(show_plot=True,save_plot=True,show_points_params=True):
    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    make_plot(df = df,
              min_or_max = "max", 
              variable = "max_amplitude", 
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)


def compare_obv_thr_pixels_std(show_plot=True,save_plot=True,show_points_params=True):
    def adapted_l2(vector) :
        vector = list(vector)
        new_vector = []
        for i in range(len(vector)) :
            x_str = vector[i].split('\n ')
            x = []
            x.append(float(x_str[0][2:-1]))
            x.append(float(x_str[1][1:-2]))
            new_vector.append(np.sqrt(x[0]**2+x[1]**2))
        return new_vector

    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    df["above-thld_pixels_std"] = adapted_l2(df["above-thld_pixels_std"])
    make_plot(df = df,
              min_or_max = "min", 
              variable = "above-thld_pixels_std", 
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)


def compare_obv_thr_pixels(show_plot=True,save_plot=True,show_points_params=True):
    def count_pixels(pixels) :
        pixels = list(pixels)
        new_pixels = []
        for i in range(len(pixels)) :
            p = pixels[i].replace("[",'')
            p = p.replace("]",'')
            p = p.replace("\n",' ')
            p = re.sub(r'\s(?:\s)+', ' ', p)
            p = p.split((' '))
            new_pixels.append(len(p)//2)
        return new_pixels

    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    df["above-thld_pixels"] = count_pixels(df["above-thld_pixels"])
    make_plot(df=df,
              min_or_max="min",
              variable="above-thld_pixels",
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)

def compare_barycenter_location(show_plot=True, save_plot=True, show_points_params=True):
    def adapted_l2(vector) :
        vector = list(vector)
        new_vector = []
        for i in range(len(vector)) :
            x_str = str(vector[i])
            if x_str != "nan" :
                x_str = re.sub(r'(\[|\])', '', x_str)
                x_str = re.sub(r'[\s]+', ' ', x_str)
                x_str = x_str.split(' ')
                x = []
                x.append(float(x_str[0])-24.5)
                x.append(float(x_str[1])-24.5)
                new_vector.append(np.sqrt(x[0]**2+x[1]**2))
            else :
                new_vector.append(None)
        return new_vector

    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    df["barycenter_location"] = adapted_l2(df["barycenter_location"]) #/!\ here we have the true average equal to (24.5,24.5) and not (25,25) due to the rounding as integer. See generate_data_noise_grading.py line 60.
    make_plot(df=df,
              min_or_max="min",
              variable="barycenter_location",
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)
   
def make_plot(df, 
              min_or_max, 
              variable, 
              show_plot=True, 
              save_plot=True, 
              show_points_params=True):

    df = df[df['noise_level'] != 0.05]

    sub_df = pd.DataFrame(columns=["noise_level",
                                   "algorithm", 
                                   variable])

    to_be_droped = list({"barycenter_location", 
                         "above-thld_pixels", 
                         "above-thld_pixels_std", 
                         "max_amplitude"} - {variable})

    for noise_lvl in df["noise_level"].unique() :
        sub_noise_df = df[df["noise_level"] == noise_lvl]

        for algo in sub_noise_df["algorithm"].unique() :
            sub_algo_noise_df = sub_noise_df[sub_noise_df["algorithm"] == algo]

            if min_or_max == "min" :
                maxminimum = sub_algo_noise_df.min()[variable]
            elif min_or_max == "max" :
                maxminimum = sub_algo_noise_df.max()[variable]

            sub_df = sub_df.append(df.loc[df[variable] == maxminimum].drop(to_be_droped, axis=1))

    sub_df.drop_duplicates(inplace=True)
    sub_df.sort_values(by=["noise_level"],inplace=True)

    nb_algo = len(sub_df["algorithm"].unique())

    plt.figure(1,figsize=(15,1+2.7*nb_algo))
    colors = ["red", "green", "blue", "magenta", "orange", "cyan"]
    i=0
    for algo in sub_df["algorithm"].unique() :
        plt.subplot(nb_algo,1,i+1)
        sub_algo_variable_df = pd.DataFrame(sub_df[sub_df['algorithm'] == algo])
        sub_algo_variable_df.dropna(axis=1, how='any', inplace=True)
        x = sub_algo_variable_df["noise_level"]
        y = sub_algo_variable_df[variable]
        label = algo
        plt.plot(x, y,label=label, color=colors[i], marker='D', alpha=.5)

        #print the parameters on the plot
        if show_points_params :
            j=0
            texts=[]
            for noise_lvl in sub_algo_variable_df["noise_level"].unique() :
                sub_algo_noise_variable_df = pd.DataFrame(sub_algo_variable_df[sub_algo_variable_df['noise_level'] == noise_lvl])
                text_param = get_label(sub_algo_noise_variable_df.drop(['noise_level',
                                                                        'algorithm'], axis='columns'))
                s = text_param
                x = noise_lvl
                y = sub_algo_noise_variable_df[variable].iloc[0]
                t = plt.text(x=x+0.01, 
                             y=y, 
                             s=text_param, 
                             color=colors[i], 
                             rotation=0)
                t.set_bbox(dict(facecolor='white', 
                                alpha=0.5, 
                                edgecolor='gray',
                                boxstyle='round'))
                texts.append(t)
                j+=1
        i+=1
        adjust_text(texts)
        plt.legend()
        plt.ylabel(variable)

    plt.xlabel("noise_level")

    title = "Compare {} {}".format("maximum" if min_or_max == 'max' else "minimum",variable,)

    # plt.tight_layout()
    plt.suptitle(title + "\n\
                 noise_level in ({})".format(', '.join([str(n) for n in df['noise_level'].unique()])), 
                 fontsize=15.0, fontweight='bold')
    if save_plot:
        plt.savefig("./results/"+title+".png")
    if show_plot:
        plt.show()
    plt.close()



if __name__ == "__main__":
    # collect("../test_algos/results")
    # collect("../../deliverable_clean/test_algos/results")
    # compare_max_amplitude(show_plot=False)
    # compare_obv_thr_pixels_std(show_plot=False)
    # compare_obv_thr_pixels(show_plot=False)
    compare_barycenter_location(show_plot=True)
