import os
from time import time

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import ot
import ot.plot

from poissonblending import blend


def im2mat(img):
    """Converts an image to a matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts a matrix back to an image"""
    return X.reshape(shape)


def color_transf(file1="art1.png", file2="art2.png", show_pixel_value=True, random_image=True):
    r = np.random.RandomState(42)

    print("Getting the images...")
    t = time()
    if random_image:
        s = 4
        img1 = r.random(size=(s, s, 3))
        # img1[:,:,1] = 0
        img2 = r.random(size=(s, s, 3))
        # img2[:,:,1] = 0
    else:
        CURR_DIR = os.path.dirname(os.path.realpath(__file__))
        print("Getting the images in the directory :",CURR_DIR + "/data/")
        img1 = pl.imread(f"{CURR_DIR}/data/{file1}").astype(np.float64)[:, :, :3]
        img2 = pl.imread(f"{CURR_DIR}/data/{file2}").astype(np.float64)[:, :, :3]

    if img1.size//3 > 100 :
        show_pixel_value = False

    plt.figure(1, figsize=(10, 10))
    plt.suptitle("Original Images", fontsize=14)
    ###########################
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title("Image1")
    if show_pixel_value:
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                text = img1[i, j].round(2)
                text = f"R : {text[0]}\nV : {text[1]}\nB : {text[2]}"
                plt.text(j, i, text, color="white", fontsize=8, fontweight="bold", ha="center", va="center")

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title("Image2")
    if show_pixel_value:
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                text = img2[i, j].round(2)
                text = f"R : {text[0]}\nV : {text[1]}\nB : {text[2]}"
                plt.text(j, i, text, color="white", fontsize=8, fontweight="bold", ha="center", va="center")
    ###########################
    print(f"DONE. elapsed time = {round(time() - t, 3)}s.\n")

    Image1 = im2mat(img1)
    Image2 = im2mat(img2)


    n = min(1000, Image1.shape[0])

    if n < Image1.shape[0]:
        print(f"Sampling {n} points...")
        t = time()
        idx1 = r.randint(Image1.shape[0], size=(n,))
        idx2 = r.randint(Image2.shape[0], size=(n,))

        Xs = Image1[idx1, :]
        Xt = Image2[idx2, :]
    else:
        print(f"Getting all the {n} points...")
        Xs = Image1
        Xt = Image2

    ###########################
    plt.subplot(2, 2, 3)
    plt.scatter(Xs[:, 0], Xs[:, 2], c=Xs)
    plt.xlabel("red")
    plt.ylabel("blue")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("Image1 distribution")

    plt.subplot(2, 2, 4)
    plt.scatter(Xt[:, 0], Xt[:, 2], c=Xt)
    plt.xlabel("red")
    plt.ylabel("blue")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("Image2 distribution")
    ###########################
    print(f"DONE. elapsed time = {round(time() - t, 3)}s.\n")

    print("Fitting EMD transport...")
    t = time()
    ot_emd = ot.da.EMDTransport(metric="cityblock")
    ot_emd.fit(Xs=Xs, Xt=Xt)
    print(f"DONE. elapsed time = {round(time() - t, 3)}s.\n")

    print("Transporting colors from one image to the other...")
    t = time()
    trans_Xt_emd = ot_emd.inverse_transform(Xt=Image2)
    trans_Xs_emd = ot_emd.transform(Xs=Image1)

    trans_Xt_emd_sample = ot_emd.inverse_transform(Xt=Xt)
    trans_Xs_emd_sample = ot_emd.transform(Xs=Xs)

    img1t = mat2im(trans_Xs_emd, img1.shape)
    img2t = mat2im(trans_Xt_emd, img2.shape)

    plt.figure(2, figsize=(10, 10))
    plt.suptitle("Transformed Images", fontsize=14)
    ###########################
    plt.subplot(2, 2, 1)
    plt.imshow(img1t)
    plt.axis('off')
    plt.title("Image1_transf")
    if show_pixel_value:
        for i in range(img1t.shape[0]):
            for j in range(img1t.shape[1]):
                text = img1t[i, j].round(2)
                text = f"R : {text[0]}\nV : {text[1]}\nB : {text[2]}"
                plt.text(j, i, text, color="white", fontsize=8, fontweight="bold", ha="center", va="center")

    plt.subplot(2, 2, 2)
    plt.imshow(img2t)
    plt.axis('off')
    plt.title("Image2_transf")
    if show_pixel_value:
        for i in range(img2t.shape[0]):
            for j in range(img2t.shape[1]):
                text = img2t[i, j].round(2)
                text = f"R : {text[0]}\nV : {text[1]}\nB : {text[2]}"
                plt.text(j, i, text, color="white", fontsize=8, fontweight="bold", ha="center", va="center")
    ###########################
    print(f"DONE. elapsed time = {round(time() - t, 3)}s.\n")

    ###########################
    plt.subplot(2, 2, 3)
    plt.scatter(trans_Xs_emd_sample[:, 0], trans_Xs_emd_sample[:, 2], c=trans_Xs_emd_sample)
    plt.xlabel("red")
    plt.ylabel("blue")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("Image1_transf distribution")

    plt.subplot(2, 2, 4)
    plt.scatter(trans_Xt_emd_sample[:, 0], trans_Xt_emd_sample[:, 2], c=trans_Xt_emd_sample)
    plt.xlabel("red")
    plt.ylabel("blue")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("Image2_transf distribution")
    ###########################

    img1_rb = np.delete(Xs, 1, 1)
    img2_rb = np.delete(Xt, 1, 1)
    m = ot.dist(Xs, Xt)

    ot_emd = ot.emd(np.ones(shape=(Xs.shape[0])), np.ones(shape=(Xt.shape[0])), m)
    w = sum(sum(ot_emd * m))

    plt.figure(3, figsize=(10, 5))
    plt.suptitle(f"Optimal Transport with W = {round(w,3)}", fontsize=14)
    ###########################
    plt.subplot(1, 2, 1)
    if n <= 32 :
        plt.scatter(img1_rb[:, 0], img1_rb[:, 1], color="magenta", label='Image1')
        plt.scatter(img2_rb[:, 0], img2_rb[:, 1], color="cyan", label='Image2')
        ot.plot.plot2D_samples_mat(img1_rb, img2_rb, m)
    else :
        plt.text(.5, .5, f"n = {n} - Too many points.", color="red", fontsize=10, fontweight="bold", ha="center", va="center")
    plt.xlabel("red")
    plt.ylabel("blue")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("Distance")

    plt.subplot(1, 2, 2)
    if n <= 100 :
        plt.scatter(img1_rb[:, 0], img1_rb[:, 1], color="magenta", label='Image1')
        plt.scatter(img2_rb[:, 0], img2_rb[:, 1], color="cyan", label='Image2')
        for i in range(ot_emd.shape[0]):
            for j in range(ot_emd.shape[1]):
                x = [img1_rb[i, 0], img2_rb[j, 0]]
                y = [img1_rb[i, 1], img2_rb[j, 1]]
                pl.plot(x, y, color="black", alpha=ot_emd[i, j] / ot_emd.max())
        pl.legend()
    else:
        plt.text(.5, .5, f"n = {n} - Too many points.", color="red", fontsize=10, fontweight="bold", ha="center", va="center")
    plt.xlabel("red")
    plt.ylabel("blue")
    plt.title("Transport")
    ###########################
    pl.show()


def face_swap():
    img_mask = plt.imread("./data/mask.png")[:, :, :3]
    img_source = plt.imread("./data/source.png")[:, :, :3]
    img_target = plt.imread("./data/target.png")[:, :, :3]

    nbsample = 500
    off = (35, -15)

    img_ret1 = blend(img_target, img_source, img_mask, offset=off)
    img_ret3 = blend(img_target, img_source, img_mask, reg=5, eta=1, nbsubsample=nbsample, offset=off, adapt='linear')
    img_ret4 = blend(img_target, img_source, img_mask, reg=5, eta=1, nbsubsample=nbsample, offset=off, adapt='kernel')

    # %%
    fs = 30
    f, axarr = pl.subplots(1, 5, figsize=(30, 7))
    newax = f.add_axes([0.15, 0, 0.32, 0.32], anchor='NW', zorder=1)
    newax.imshow(img_mask)
    newax.axis('off')
    newax.set_title('mask')
    axarr[0].imshow(img_source)
    axarr[0].set_title('Source', fontsize=fs)
    axarr[0].axis('off')
    axarr[1].imshow(img_target)
    axarr[1].set_title('target', fontsize=fs)
    axarr[1].axis('off')
    axarr[2].imshow(img_ret1)
    axarr[2].set_title('[Perez 03]', fontsize=fs)
    axarr[2].axis('off')
    axarr[3].imshow(img_ret3)
    axarr[3].set_title('Linear', fontsize=fs)
    axarr[3].axis('off')
    axarr[4].imshow(img_ret4)
    axarr[4].set_title('Kernel', fontsize=fs)
    axarr[4].axis('off')
    pl.subplots_adjust(wspace=0.1)
    pl.tight_layout()
    pl.show()


if __name__ == "__main__":
    color_transf()


