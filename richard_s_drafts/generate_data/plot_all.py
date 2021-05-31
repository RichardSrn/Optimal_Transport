#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join


onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
onlyfiles = [file for file in onlyfiles if file[-4:] == ".npy"]
onlyfiles.sort()

size_x, size_y = 6,6

plt.figure(1, figsize=(25,25))
for i in range(len(onlyfiles)) :
    img = np.load("./data/"+onlyfiles[i])[0]
    plt.subplot(size_x,size_y,i+1)
    plt.title(onlyfiles[i][16:-4], fontsize=5)
    plt.axis('off')
    plt.imshow(img)
plt.tight_layout()
plt.show()