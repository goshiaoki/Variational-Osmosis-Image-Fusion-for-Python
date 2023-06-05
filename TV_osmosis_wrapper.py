import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
import time
from block_ipiano import *


def TV_osmosis_wrapper(foreground, background, alpha, params):
    # initialisation of u0,v0
    if params['flag_initialisation'] == 0:
        u = foreground
    elif params['flag_initialisation'] == 1:
        u = alpha * foreground + (1 - alpha) * background
    elif params['flag_initialisation'] == 2:
        mf = alpha * foreground
        mf = np.sum(mf) / np.sum(alpha)
        mb = (1 - alpha) * background
        mb = np.sum(mb) / np.sum(1 - alpha)
        u = alpha * mf + (1 - alpha) * mb
    u0 = u
    v0 = (foreground ** alpha) * (background ** (1 - alpha))

    print('-------------------------------------------------------------------------')

    # single cycle - block iPiano with backtracking
    u, v, L1, L2, E = block_ipiano(foreground, background, alpha, u0, v0, params)

    params['L1'] = L1  # keep track of the already computed L
    params['L2'] = L2  # keep track of the already computed L

    if params['plot_figures']:
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.imshow(u - params['offset'], vmin=0, vmax=1)
        plt.title('u')
        plt.subplot(2, 2, 2)
        plt.imshow(v - params['offset'], vmin=0, vmax=1)
        plt.title('v')
        plt.subplot(2, 2, 3)
        plt.imshow(u - u0)
        plt.title('u-u0')
        plt.pause(1)

    return u, v, E
