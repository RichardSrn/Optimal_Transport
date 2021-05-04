import matplotlib.pyplot as plt

from barycenter import timer
from PIL import Image, ImageDraw
import os

@timer
def images_to_gif(path = "/results/barycenter/", files_ext = ".jpg"):
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    path = CURR_DIR+path
    images = []
    for i in range(1,3) :
        img = Image.open(path+str(i)+files_ext)
        images.append(img)

    images[0].save('2D_norm_distr_reg_tuning.gif',
                   save_all=True, append_images=images[1:],
                   optimize=False, duration=10, loop=0)



if __name__ == "__main__":
    images_to_gif(path = "/results/barycenter/", files_ext = ".jpg")

