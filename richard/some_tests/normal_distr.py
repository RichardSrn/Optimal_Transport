import numpy
import numpy as np  # always need it
import ot  # ot
import ot.plot
import pylab as pl  # do the plots
import seaborn as sns
from ot.datasets import make_1D_gauss as gauss
from ot.datasets import make_2D_samples_gauss as ggauss


def normal_1D():
    n = 100
    x = np.arange(n, dtype=np.float64)

    mu_s = gauss(n, m=20, s=5)
    mu_t = gauss(n, m=60, s=10)

    pl.figure(1, figsize=(6.4, 3))
    pl.plot(x, mu_s, 'b', label='Source distribution')
    pl.plot(x, mu_t, 'r', label='Target distribution')
    pl.legend()
    pl.show()

    # loss matrix
    m = ot.dist(x.reshape((n,1)),x.reshape((n,1)))
    # m = ot.dist(mu_s.reshape((n, 1)), mu_t.reshape((n, 1)))
    m = m / m.max()

    pl.figure(2, figsize=(5, 5))
    ot.plot.plot1D_mat(mu_s, mu_t, m, 'Distance matrix m')
    pl.show()

    # Earth Mover’s Distance
    emd = ot.emd(mu_s, mu_t, m)
    pl.figure(3, figsize=(5, 5))
    ot.plot.plot1D_mat(mu_s, mu_t, emd, title='OT emd matrix')
    pl.show()

    # Sinkhorn algorithm
    lambd = 1e-3
    sink = ot.sinkhorn(mu_s, mu_t, m, lambd, verbose=True)
    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(mu_s, mu_t, sink, 'OT matrix Sinkhorn')
    pl.show()

    def f(G):
        return 0.5 * np.sum(G ** 2)

    def df(G):
        return G

    reg1 = 1e-3
    reg2 = 1e-1

    gel2 = ot.optim.gcg(mu_s, mu_t, m, reg1, reg2, f, df, verbose=True)

    pl.figure(5, figsize=(5, 5))
    ot.plot.plot1D_mat(mu_s, mu_t, gel2, 'OT entropic + matrix Frob. reg')
    pl.show()


def normal_2D_1():
    """
    Plot of points normally distributed (mu,sigma).
    Red and Green points represent two normal distribution.
    Black lines represent W distance between each points.
    """
    n = 6
    x = np.arange(n, dtype=np.float64)

    mu_s = ggauss(n=n, m=(-1, 2), sigma=np.array([[3, 0],
                                                  [-5, 3]]),
                  random_state=42)
    mu_t = ggauss(n=n, m=(2, -3), sigma=np.array([[5, 5],
                                                  [0, 2]]),
                  random_state=42)
    m = ot.dist(mu_s, mu_t)
    m = m / m.max()

    pl.scatter(mu_s[:, 0], mu_s[:, 1], color="green", alpha=.5)
    pl.scatter(mu_t[:, 0], mu_t[:, 1], color="red", alpha=.5)
    ot.plot.plot2D_samples_mat(mu_s, mu_t, m)
    pl.show()


def normal_2D_2():
    """
    Plots of 2 normal distribution.
    A plot as grid to represent the space and the hue to represent the value of a given point
    (according to the corresponding normal distribution).
    The map represent the transport matrix from mu_s to mu_t according to
    Sinkhorn algorithm with a regularization > 0.
    """
    def gggauss(xx, yy, mu_x, mu_y, sigma_x, sigma_y, rho):
        denom = 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2)
        Y = numpy.zeros(shape=(xx.shape[0], xx.shape[1]))
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x = np.array([xx[i, j]])
                y = np.array([yy[i, j]])
                a = ((x - mu_x) / sigma_x) ** 2
                b = 2 * rho * (x - mu_x) / sigma_x * (y - mu_y) / sigma_y
                c = ((y - mu_y) / sigma_y) ** 2
                Y[i, j] = np.exp(-1 / (2 * (1 - rho ** 2)) * (a - b + c)) / denom
        return Y

    def coord(xx, yy):
        c = np.zeros(shape=(xx.shape[0] * xx.shape[1], 2))
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                c[xx.shape[0] * i + j] = [xx[i, j], yy[i, j]]
        return c

    n = 25
    x = np.linspace(0, 5, n, dtype=np.float64)
    y = np.linspace(0, 5, n, dtype=np.float64)

    xx, yy = np.meshgrid(x, y)
    mu_s = gggauss(xx=xx, yy=yy, mu_x=2, mu_y=2, sigma_x=1, sigma_y=1, rho=0)
    mu_t = gggauss(xx=xx, yy=yy, mu_x=4, mu_y=4, sigma_x=1, sigma_y=1, rho=0)

    coor = coord(xx, yy)
    xx = np.hstack(xx)
    yy = np.hstack(yy)
    mu_s = np.hstack(mu_s)
    mu_t = np.hstack(mu_t)

    m = ot.dist(coor, coor)
    m = m / m.max()

    pl.figure(1, figsize=(10, 10))
    pl.subplot(2,2,3)
    sns.scatterplot(x=xx, y=yy, hue=mu_s,
                    palette=sns.color_palette("crest", as_cmap=True))
    pl.title("Norm. Distr. 1")
    # pl.legend().remove()
    pl.legend(loc="upper right")
    pl.axis("off")

    pl.subplot(2,2,2)
    sns.scatterplot(x=xx, y=yy, hue=mu_t,
                    palette=sns.color_palette("flare", as_cmap=True))
    pl.title("Norm. Distr. 2")
    # pl.legend().remove()
    pl.legend(loc="lower left")
    pl.axis("off")

    reg = 0.1
    sinkhorn = ot.sinkhorn(mu_s, mu_t, reg=reg, M=m)

    # pl.figure(2, figsize=(10, 10))
    pl.subplot(2,2,4)
    im = pl.imshow(sinkhorn)
    pl.axis("off")
    # for i in range(len(mu_s)):
    #     for j in range(len(mu_t)):
    #         pl.text(j, i, np.round(sinkhorn[i, j], 3),
    #                        ha="center", va="center", color="w")
    pl.title('Transport matrix \n Sinkhorn - reg ='+str(reg))
    pl.xlabel('mu_t')
    pl.ylabel('mu_s')
    pl.show()
