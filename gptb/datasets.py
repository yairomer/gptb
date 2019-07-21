import os
import struct

import numpy as np


def load_mnist(path, train=True):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    prefix = 'train' if train else 't10k'

    ## Load images:
    filename = os.path.join(path, '{}-images-idx3-ubyte'.format(prefix))
    with open(filename, 'rb') as fid:
        # magic, num, rows, cols = struct.unpack(">IIII", fid.read(16))
        _, _, rows, cols = struct.unpack(">IIII", fid.read(16))
        images = np.fromfile(fid, dtype=np.uint8).reshape(-1, rows, cols)

    ## Load labels:
    filename = os.path.join(path, '{}-labels-idx1-ubyte'.format(prefix))
    with open(filename, 'rb') as fid:
        # magic, num = struct.unpack(">II", fid.read(8))
        _, _ = struct.unpack(">II", fid.read(8))
        labels = np.fromfile(fid, dtype=np.int8)

    return images, labels
