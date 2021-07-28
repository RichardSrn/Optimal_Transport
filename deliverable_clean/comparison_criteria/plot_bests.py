#! /usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def get_files(path="./results"):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyfiles = [file for file in onlyfiles if (file[:4] == "Logs" and file[-4:] == ".csv")]
    onlyfiles.sort()

    return onlyfiles


def plot_bests(path="./results", show=False, save=True) :
    files = get_files(path)
    barys = np.zeros(shape=(4,5,50,50))
    for file in files :
        df = pd.read_csv(os.path.join(path,file))
        variable = file[10:-4]
        algos = sorted(df["algorithm"].unique())
        noise_lvls = sorted(df["noise_level"].unique())
        
        plt.figure(1,figsize=(13,8))

        i = 0
        for algo in algos :
            df_algo = df[df["algorithm"] == algo].dropna(axis=1).drop_duplicates()
            df_algo_noise = pd.DataFrame(columns = df_algo.columns)
            for nl in noise_lvls :
                df_algo_noise = df_algo_noise.append(df_algo[df_algo["noise_level"]==nl].iloc[0], ignore_index=True)
            del(df_algo)

            file_names = []
            for j in range(len(noise_lvls)) :
                file_names.append(df[(df["algorithm"]==algo) & (df["noise_level"]==noise_lvls[j])]["file_name"].iloc()[0])
                barys[i,j] = np.load(os.path.join("../test_algos/results/",file_names[j]))

            for j in range(5) :
                file_name = file_names[j]
                plt.subplot(4, 5, i*5+j+1)
                plt.imshow(barys[i,j])
                rounded_amplitude = "{:e}".format(df_algo_noise['max_amplitude'].iloc[j])
                rounded_amplitude = rounded_amplitude[:4]+rounded_amplitude[-4:]
                name = file_name.split('/')[1][20:-4]
                name = re.sub(r"_intensity_\w+", "", name)
                name = re.sub(r"_outer-inner_\d+-\d+", "", name)
                name = re.sub(r"_c_-\d+.\d+", "", name)
                name = re.sub(r"_sample_\d", "", name)
                name = re.sub(r"_samples_\d", "", name)
                name = re.sub(r"^_+|_+$", "", name)
                plt.title(f"{name}")#, fontsize=10)
                if j == 0:
                    plt.ylabel(str("\n".join(algo.split(" "))+f"\n\namplitude:{rounded_amplitude}"), fontweight="bold")
                else :
                    plt.ylabel(f"amplitude:{rounded_amplitude}", fontweight="bold")
                if i == 3:
                    plt.xlabel(f"noise level = {noise_lvls[j]}", fontweight="bold")
                plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
            i+=1

        plt.tight_layout()
        # plt.suptitle(f"Plot one of best barycenters for each\nnoise level and algorithm - variable : {variable}")
        if show: 
            plt.show()
        if save :
            plt.savefig(f"./results/plot_best_for_{variable}.png")
        plt.close()


if __name__ == "__main__":
    plot_bests()