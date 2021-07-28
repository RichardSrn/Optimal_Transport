#! /usr/bin/env python3

from os import listdir
from os.path import join, isfile
import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_csv(path="./data"):
    """
    Get the csv_time files.
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [file for file in files if re.search(r"(csv)$", "times_ENT.csv")]

    csv = {file : [] for file in files}
    for file in files :
        csv[file] = pd.read_csv(join(path,file))
        csv[file].sort_values(by = csv[file].columns[0], inplace=True)
    return csv

def compare_time(show=False, save=True):
    csv = get_csv(path = "../test_algos/results")

    # plot_id = 1
    for file,df in csv.items():
        fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(2,2,plot_id,projection="3d")
        # plot_id+=1
        ax = plt.axes(projection="3d")

        columns = df.columns
        x = df[columns[0]]
        y = df[columns[1]]
        i=0
        for col in columns[:2] :
            unique = df[col].unique()
            if len(unique) == 2 :
                if i == 0:
                    x = [0 if m == unique[0] else 1 for m in df[col]]
                    ax.set_xticks(ticks=[0,1], labels=unique)
                    ax.set_xticklabels(labels=unique, rotation=45, fontsize=8)
                elif i == 1:
                    y = [0 if m == unique[0] else 1 for m in df[col]]
                    ax.set_yticks(ticks=[0,1])
                    ax.set_yticklabels(labels=unique, rotation=0, fontsize=8)
            i+=1
        
        ax.scatter(x, y, df["time"], c=df["time"])

        # ax.set_title(file[:-4])

        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_zlabel(columns[2], rotation=90)
        if show :
            plt.show()
        if save :
            plt.savefig("./results"+"/"+str(file[:-4]))




if __name__ == '__main__':
    compare_time()