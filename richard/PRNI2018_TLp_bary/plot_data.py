import math

import matplotlib.pyplot as plt
import numpy as np


def main():
    # rng = np.random.RandomState(42)
    #
    # n = 9
    #
    # index = rng.choice(np.arange(200), size=(n), replace=False)
    # index.sort()
    #
    # data = np.load("./artificial_data.npy")[index, :, :]
    #
    # plt.figure(1, figsize=(10, 10))
    # for i in range(n):
    #     plt.subplot(math.floor(np.sqrt(n)), math.ceil(np.sqrt(n)), i + 1)
    #     plt.imshow(data[i, :, :])
    #     plt.title("Element indexed {}".format(index[i]))
    # plt.suptitle("Visualization of {}, randomly chosen, elements of the artificial_data file.".format(n))
    # plt.show()

    #plot barycenters
    bary_KBCM = np.load("./bary_KBCM.npy").reshape((50,50))
    bary_TLp = np.load("./bary_TLp.npy").reshape((50,50))
    barys_TLp = np.load("./barys_TLp.npy").reshape((10,50,50))

    plt.figure(2, figsize=(10,10))
    plt.subplot(4,3,1)
    plt.imshow(bary_KBCM)
    plt.title("Bary_KBCM")
    plt.axis('off')

    plt.subplot(4,3,2)
    plt.imshow(bary_TLp)
    plt.title("Bary_TLP")
    plt.axis("off")

    for i in range(10):
        plt.subplot(4,3,i+3)
        plt.imshow(barys_TLp[i])
        plt.title("Barys_TLp number {}".format(i))
        plt.axis("off")

    plt.suptitle("Barycenters from Barycenter_example")
    plt.show()




if __name__ == "__main__":
    main()
