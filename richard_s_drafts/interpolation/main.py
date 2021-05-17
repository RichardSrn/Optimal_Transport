import numpy as np
from const import ART_DATA
from hist_from_images import hist_from_images
from coupling_from_2_images import coupling_from_2_images
from barycenter_from_coupling import barycenter_from_coupling


def main() :
    #define the rng
    rng = np.random.RandomState(42)

    #choose 2 images
    index = rng.choice(np.arange(199), 2, replace=False)
    img1, img2 = np.load(ART_DATA)[index]

    #turn 2D images into 1D vector --histogram--
    hist1 = hist_from_images(img1)
    hist2 = hist_from_images(img2)

    #get the coupling matrix
    coupl = coupling_from_2_images(hist1, hist2)

    #get the barycenter
    barycenter = barycenter_from_coupling(hist1, hist2, coupl)


if __name__ == "__main__" :
    main()