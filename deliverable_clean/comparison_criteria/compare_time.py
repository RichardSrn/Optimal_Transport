#! /usr/bin/env python3

from os import listdir
from os.path import join, isfile
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def compare_time():
    csv = get_csv(path = "../test_algos/results")
    
    for file,df in csv.items():
        pass




if __name__ == '__main__':
    compare_time()