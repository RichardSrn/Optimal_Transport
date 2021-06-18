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
    """
    Get the description of each point based on the data frame.
    If multiple parameters give the same output, the parameter p is denoted as :
        p = a,b,c
    a,b,c being the possible values of the parameter.
    """
    def round_sig(x, sig=2):
        """Round x to a certain amount of significant digits."""
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
                        values[i] = re.sub(r"(0+e)","e",values[i])
                        values[i] = re.sub(r"(\.e)", "e", values[i])
                        values[i] = re.sub(r"(\.0+)", "", values[i])
            except :
                pass
            finally :
                values[i] = str(values[i])

        if c in shorten.keys():
            text.append(shorten[c]+'='+", ".join(set(values))+"")
        else :
            text.append(c+'='+", ".join(set(values))+"")
    # text ="\n".join( text )
    return text


def get_title_and_text(list_text, plot_title) :
    """
    If a parameter's value is the same for every point, add it to the title.
    This allows to save some space on the plot.
    """
    set_text = set(list_text[0])
    for l in list_text[1:] :
        set_text = set_text & set(l)
    if len(set_text) != 0 :
        plot_title = plot_title + " - " + "; ".join(list(set_text))
        for s in set_text :
            for i in range(len(list_text)) :
                if s in list_text[i] :
                    list_text[i].remove(s)

    for i in range(len(list_text)) :
        list_text[i] = "\n".join(list_text[i])

    return (list_text,plot_title)


def parse_key(key) :
    """
    Change the dictionary's keys to shorter or more descriptive keys.
    """
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
    """
    Get the barycenters files.
    """
    directories = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    directories.remove('noisefrees_control')

    tree = {d : [] for d in directories}

    for d in directories :
        tree[d] = [f for f in os.listdir(os.path.join(path,d)) if os.path.isfile(os.path.join(path,d,f))]
        tree[d] = [file for file in tree[d] if file[-4:] == ".npy"]
        tree[d].sort()

    return tree


def collect(path = "./data/", save=True, save_path="./results/"):
    """
    Collect the data from the barycenter's files.
    Then summarize everything in a DataFrame and save it (if save==True) as a file.
    """
    tree = get_files(path)
    data = {d : np.array([np.load(os.path.join(path,d,file)) for file in tree[d]]) for d in tree.keys()}
    
    data = standardize.main(data)

    df = pd.DataFrame(columns=[ "noise_level",
                                "max_amplitude",
                                "barycenter_dist",
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

            max_ampl, barycenter_loc, pixels, std = criteria(barycenter)
            param["max_amplitude"] = max_ampl
            param["barycenter_dist"] = barycenter_loc
            param["above-thld_pixels"] = pixels
            param["above-thld_pixels_std"] = std
            df = df.append(param, ignore_index=True)

    if save :
        df.to_csv(os.path.join(save_path, "DataFrame_summary.csv"), index=False)

    return df


def compare_max_amplitude(show_plot=True,
                          save_plot=True,
                          show_points_params=True):
    """
    Compare the max amplitude of the barycenter's for each algorithms.
    """
    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    make_plot(df = df,
              min_or_max = "max", 
              variable = "max_amplitude", 
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)


def compare_obv_thr_pixels_std(show_plot=True,
                               save_plot=True,
                               show_points_params=True):
    """
    Compare the standard deviation of the pixels above threshold (threshold = 1/2 * max_amplitude).
    Get the minimum std for each algorithm.
    The smallest activation zone is better as we want a sharp barycenter.
    """
    def adapted_l2(vector) :
        vector = list(vector)
        new_vector = []
        for i in range(len(vector)) :
            x_str = str(vector[i])
            if x_str != "nan" :
                x_str = re.sub(r'(\[|\])', '', x_str)
                x_str = re.sub(r'(\s+)', ' ', x_str)
                x_str = re.sub(r'((?:\s)+)$', '', x_str)
                x_str = x_str.split(' ')
                x = []
                x.append(float(x_str[0]))
                x.append(float(x_str[1]))
                new_vector.append(np.sqrt(x[0]**2+x[1]**2))
            else :
                new_vector.append(None)
        return new_vector

    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    df["above-thld_pixels_std"] = adapted_l2(df["above-thld_pixels_std"])
    make_plot(df = df,
              min_or_max = "min", 
              variable = "above-thld_pixels_std", 
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)


def compare_obv_thr_pixels(show_plot=True,
                           save_plot=True,
                           show_points_params=True):
    """
    Compare the number of pixels above threshold (threshold = 1/2 * max_amplitude).
    The smallest number of pixels above threshold is best as we want a sharp barycenter.
    """
    def count_pixels(pixels) :
        pixels = list(pixels)
        new_pixels = []
        for i in range(len(pixels)) :
            if str(pixels[i]) != "nan" :
                p = pixels[i].replace("[",'')
                p = p.replace("]",'')
                p = p.replace("\n",' ')
                p = re.sub(r'\s(?:\s)+', ' ', p)
                p = p.split((' '))
                new_pixels.append(len(p)//2)
            else :
                new_pixels.append(None)
        return new_pixels

    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    df["above-thld_pixels"] = count_pixels(df["above-thld_pixels"])
    make_plot(df=df,
              min_or_max="min",
              variable="above-thld_pixels",
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)


def compare_barycenter_dist(show_plot=True, 
                                save_plot=True, 
                                show_points_params=True):
    """
    Compare the barycenter's distance to the image's center.
    We generate the data so that the barycenters are determined to be in the center of the image (24.5,24.5)
    as long as we compute it on the n'th first images, n even.
    We then compare the smallest l2 norm between the barycenter's center and the image's center.
    The barycenter's center is computed as the weighted average of all the above threshold pixels.
    """
    def adapted_l2(vector) :
        vector = list(vector)
        new_vector = []
        for i in range(len(vector)) :
            x_str = str(vector[i])
            if x_str != "nan" :
                x_str = re.sub(r'(\[|\])', '', x_str)
                x_str = re.sub(r'[\s]+', ' ', x_str)
                x_str = re.sub(r'((?:\s)+)$', '', x_str)
                x_str = x_str.split(' ')
                x = []
                x.append(float(x_str[0])-24.5)
                x.append(float(x_str[1])-24.5)
                new_vector.append(np.sqrt(x[0]**2+x[1]**2))
            else :
                new_vector.append(None)
        return new_vector

    df = pd.read_csv(os.path.join("./results", "DataFrame_summary.csv"))
    df["barycenter_dist"] = adapted_l2(df["barycenter_dist"]) #/!\ here we have the true average equal to (24.5,24.5) and not (25,25) due to the rounding as integer. See generate_data_noise_grading.py line 60.
    make_plot(df=df,
              min_or_max="min",
              variable="barycenter_dist",
              show_plot=show_plot, 
              save_plot=save_plot, 
              show_points_params=show_points_params)
   

def make_plot(df, 
              min_or_max, 
              variable, 
              show_plot=True, 
              save_plot=True, 
              show_points_params=True):
    """
    Make the plot for any comparison.
    If show_points_params==True, show the parameters of the algorithms for a each point.
    """

    df = df[df['noise_level'] != 0.05]

    sub_df = pd.DataFrame(columns=["noise_level",
                                   "algorithm", 
                                   variable])

    to_be_droped = list({"barycenter_dist", 
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

    plt.figure(1,figsize=(15,1+2.9*nb_algo))
    colors = ["red", "green", "blue", "magenta", "orange", "cyan"]
    i=0
    for algo in sub_df["algorithm"].unique() :
        plt.subplot(nb_algo,1,i+1)
        sub_algo_variable_df = pd.DataFrame(sub_df[sub_df['algorithm'] == algo])
        sub_algo_variable_df.dropna(axis=1, how='any', inplace=True)
        x = sub_algo_variable_df["noise_level"]
        y = sub_algo_variable_df[variable]
        label = algo
        plot_title = algo
        plt.plot(x, y,label=label, color=colors[i], marker='D', alpha=.5)

        #print the parameters on the plot
        if show_points_params :
            j=0
            texts=[]
            s_s = []
            x_s = []
            y_s = []
            for noise_lvl in sub_algo_variable_df["noise_level"].unique() :
                sub_algo_noise_variable_df = pd.DataFrame(sub_algo_variable_df[sub_algo_variable_df['noise_level'] == noise_lvl])
                text_param = get_label(sub_algo_noise_variable_df.drop(['noise_level',
                                                                        'algorithm'], axis='columns'))
                s_s.append(text_param)
                x_s.append(noise_lvl)
                y_s.append(sub_algo_noise_variable_df[variable].iloc[0])
                j+=1

            list_text,plot_title = get_title_and_text(s_s,plot_title)
            for k in range(j) :
                t = plt.text(x=x_s[k]+0.01, 
                             y=y_s[k], 
                             s=list_text[k], 
                             color=colors[i], 
                             rotation=0)
                t.set_bbox(dict(facecolor='white', 
                                alpha=0.5, 
                                edgecolor='gray',
                                boxstyle='round'))
                texts.append(t)
        i+=1
        adjust_text(texts, precision=1e-3)
        # plt.legend()
        plt.title(plot_title, fontsize=10.0, fontweight='bold')
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

def compare_all(re_collect=True):
    """
    Run all the comparisons.
    """
    if re_collect:
        collect("../test_algos/results")
    compare_max_amplitude(show_plot=False)
    compare_obv_thr_pixels_std(show_plot=False)
    compare_obv_thr_pixels(show_plot=False)
    compare_barycenter_dist(show_plot=False)



if __name__ == "__main__":
    compare_all()