from IPython import get_ipython
ipy = get_ipython()

import sys
import os
import time
import glob
import itertools
import socket

## Matplotlib
ipy.magic("matplotlib notebook")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import PIL
import cv2
import tqdm
import skimage
import torch
import torchvision

mpl_dpi = mpl.rcParams['figure.dpi']

## Auto reload - Use %autoreload to reload packages
ipy.magic("load_ext autoreload")


## TQDM
def _tqdm_replacemnt(*args, **kwargs):
    kwargs.pop('ncols', None)
    return tqdm.tqdm_notebook(*args, **kwargs)


tqdm.tqdm = _tqdm_replacemnt

## Jupyter widgets
import ipywidgets as widgets

## Jupyter display
import IPython.display
from IPython.display import display

HTML = lambda x: display(IPython.display.HTML(x))
Markdown = lambda x: display(IPython.display.Markdown(x))
Image = lambda x: display(PIL.Image.fromarray(x))
ImageFile = lambda x: display(IPython.display.Image(x))


## Shortcut to plotimages
def imshow(img, scale=1, **kwargs):
    fig = plt.figure(figsize=(img.shape[1] / mpl_dpi * scale, img.shape[0] / mpl_dpi * scale))
    ax = fig.add_axes((0, 0, 1, 1))
    if not (img.ndim == 3 and img.shape[2] == 3):
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'gray'

    img_image = ax.imshow(img, **kwargs)

    return fig, ax, img_image
