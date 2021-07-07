#! /usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_files(path="./results"):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyfiles = [file for file in onlyfiles if (file[:4] == "Logs" and file[-4:] == ".csv")]
    onlyfiles.sort()

    return onlyfiles


def plot_bests(path="./results", show=False, save=True) :
    files = get_files(path)    
    indexes = pd.read_csv(os.path.join(path,"indexes.csv"))
    barys = np.zeros(shape=(4,5,50,50))
    for file in files :
        df = pd.read_csv(os.path.join(path,file))
        variable = file[10:-4]
        algos = sorted(df["algorithm"].unique())
        noise_lvls = sorted(df["noise_level"].unique())
        
        plt.figure(1,figsize=(15,15))

        i = 0
        for algo in algos :
            df_algo = df[df["algorithm"] == algo].dropna(axis=1).drop_duplicates()
            df_algo_noise = pd.DataFrame(columns = df_algo.columns)
            for nl in noise_lvls :
                df_algo_noise = df_algo_noise.append(df_algo[df_algo["noise_level"]==nl].iloc[0], ignore_index=True)
            
            del(df_algo)

            for j in range(len(noise_lvls)) :
                file_dir = "/".join(list(indexes.iloc[df_algo_noise["Unnamed: 0"].iloc[j]]))
                barys[i,j] = np.load(os.path.join("../test_algos/results/",file_dir))

            for j in range(5) :
                plt.subplot(4, 5, i*5+j+1)
                plt.imshow(barys[i,j])
                rounded_vmax = "{:e}".format(df_algo_noise['max_amplitude'].iloc[j])
                rounded_vmax = rounded_vmax[:4]+rounded_vmax[-4:]
                plt.title(f"vmax = {rounded_vmax}", fontsize=10)
                if j == 0:
                    plt.ylabel("\n".join(algo.split(" ")), fontweight="bold")
                if i == 3:
                    plt.xlabel(f"noise level = {noise_lvls[j]}", fontweight="bold")
                plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
            i+=1
        plt.suptitle(f"Plot one of best barycenters for each\nnoise level and algorithm - variable : {variable}")
        if show: 
            plt.show()
        if save :
            plt.savefig(f"./results/plot_best_for_{variable}.png")
        plt.close()


if __name__ == "__main__":
    plot_bests()