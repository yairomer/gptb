import sys
import os
import shutil
import time
import glob
import itertools
import socket
import pickle

jupyter_frontend =  len(sys.argv) >= 1 and bool(int(sys.argv[1]))

## Matplotlib
ipy = get_ipython()
if jupyter_frontend:
    ipy.magic("matplotlib notebook")
else:
    ipy.magic("matplotlib inline")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import PIL
import cv2
import tqdm
import skimage
import skimage.io
import torch
import torchvision
import yaml

mpl_dpi = mpl.rcParams['figure.dpi']

if not ipy is None:
    ## Auto reload - Use %autoreload to reload packages
    ipy.magic("load_ext autoreload")


if jupyter_frontend:
    ## TQDM
    def _tqdm_notebook(*args, **kwargs):
        kwargs.pop('ncols', None)
        return tqdm.tqdm_notebook(*args, **kwargs)


    tqdm.tqdm_org = tqdm.tqdm
    tqdm.tqdm = _tqdm_notebook

## Jupyter widgets
import ipywidgets as widgets

## Jupyter display
import IPython
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
    ax.set_axis_off()
    fig.canvas.draw()

    return fig, ax, img_image


## Initialize Plotly
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
plotly.offline.init_notebook_mode(connected=True)