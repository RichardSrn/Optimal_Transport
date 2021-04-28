import sys

from barycenter import barycenter, barycenter2, barycenter3
from image_manip import color_transf, face_swap
from manhattan_example import manhattan
from normal_distr import normal_1D, normal_2D_1, normal_2D_2


def main(choice=32):
    if choice == 0:
        manhattan()

    elif choice == 1:
        normal_1D()
    elif choice == 11:
        normal_2D_1()
    elif choice == 12:
        normal_2D_2()

    elif choice == 2:
        # https://towardsdatascience.com/hands-on-guide-to-python-optimal-transport-toolbox-part-2-783029a1f062
        color_transf()
    elif choice == 21:
        face_swap()

    elif choice == 3:
        # https://pythonot.github.io/auto_examples/barycenters/plot_convolutional_barycenter.html
        barycenter()
    elif choice == 31:
        barycenter2()
    elif choice == 32:
        barycenter3()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        main(choice=int(sys.argv[1]))
