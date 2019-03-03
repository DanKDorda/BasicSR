import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def one_plot(im):
    if 'numpy' not in type(im):
        im = im.detach().float().cpu().numpy()

    plt.imshow(im)


def multiplot(*ims):
    t_list = make_grid(ims)
    one_plot(t_list)

    plt.show()
