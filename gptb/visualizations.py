import os
import subprocess
import copy
import socket
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
import visdom

def start_visdom(initial_port=9960, *args, **kwargs):
    port = initial_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    success = False
    while not success:
        try:
            sock.bind(('127.0.0.1', port))
            success = True
        except:
            port += 1
    sock.close()
    print('\nRunning visdom at port {}\n'.format(port))
    cmd = ['visdom',
            '-p', str(port),
            ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(3)

    vis = visdom.Visdom(port=port, *args, **kwargs)
    return vis

class Images2MultiGrid:
    def __init__(self, data_shape, borders=None):
    
        if len(data_shape) % 2 == 0:
            data_shape = tuple(data_shape) + (1, )
        data_shape = np.array(data_shape)
        n_levels = len(data_shape) // 2
        if borders is None:
            borders = np.arange(n_levels - 1)[::-1]
        
        def add_border_and_tile(img, reps, border_width):
            img = np.concatenate((img, np.zeros((img.shape[0], border_width), dtype=bool)), axis=1)
            img = np.concatenate((img, np.zeros((border_width, img.shape[1]), dtype=bool)), axis=0)
            img = np.tile(img, reps)
            img = img[:(img.shape[0] - border_width), :(img.shape[1] - border_width)]
            return img
        
        img = np.ones((data_shape[-3], data_shape[-2]), dtype=bool)
        for i_level in range(1, n_levels):
            border_width = borders[-i_level]
            level_height = data_shape[-(i_level * 2 + 3)]
            level_width = data_shape[-(i_level * 2 + 2)]
            img = add_border_and_tile(img, (level_height, level_width), border_width)
        
        self._indices = np.repeat(img.flatten(), data_shape[-1])
        self._shape = img.shape + (data_shape[-1],)
        self._img = np.zeros(self._shape)

        indices_img = np.reshape(np.arange(np.prod(data_shape)), data_shape)
        self._shuffle_indices = indices_img.transpose(tuple(np.arange(n_levels) * 2) + tuple(np.arange(n_levels) * 2 + 1) + (n_levels * 2,)).flatten()

        self(np.repeat(np.linspace(0, 1, np.prod(data_shape[:-1])), data_shape[-1]))
    
    def __call__(self, images):
        self._img.flat[self._indices] = images.flat[self._shuffle_indices]
        return self._img
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def last_image(self):
        return self._img

class MultiGridImagesViewer:
    def __init__(self, data_shape, borders=None, scale=1):
        self._grid_builder = Images2MultiGrid(data_shape, borders)
        
        mpl_dpi = mpl.rcParams['figure.dpi']
        self._fig = plt.figure(figsize=(self._grid_builder.shape[1] / mpl_dpi * scale, 
                                        self._grid_builder.shape[0] / mpl_dpi * scale))
        self._ax = self._fig.add_axes((0, 0, 1, 1))
        if self._grid_builder.shape[-1] == 1:
            self._img_image = self._ax.imshow(self._grid_builder.last_image[..., 0])
        else:
            self._img_image = self._ax.imshow(self._grid_builder.last_image)
    
    def __call__(self, data):
        if self._grid_builder.shape[-1] == 1:
            self._img_image.set_data(self._grid_builder(data)[..., 0])
        else:
            self._img_image.set_data(self._grid_builder(data))

        self._fig.canvas.draw()

class GridOfGridOfImages:
    def __init__(self, shape,
                 sub_border_width=0,
                 border_width=1,
                 ):
    
        self._shape = shape  # (grid_height, grid_width, sub_grid_height, sub_grid_width, image_height, image_width, n_channels)
        self._sub_border_width = sub_border_width
        self._border_width = border_width
    
        self._sub_border_v = np.ones((
            self._shape[0], self._shape[1],
            self._shape[2],
            self._shape[4], self._sub_border_width,
            self._shape[6]))

        new_width = self._shape[3] * self._shape[5] + (self._shape[3] - 1) * self._sub_border_width

        self._border_v = np.ones((
            self._shape[0],
            self._shape[2],
            self._shape[4], self._border_width,
            self._shape[6]))
        
        new_width = self._shape[1] * new_width + (self._shape[1] - 1) * self._border_width

        self._sub_border_h = np.ones((
            self._shape[0],
            self._sub_border_width, new_width,
            self._shape[6]))

        new_height = self._shape[2] * self._shape[4] + (self._shape[2] - 1) * self._sub_border_width

        self._border_h = np.ones((
            self._border_width, new_width,
            self._shape[6]))

        new_height = self._shape[0] * new_height + (self._shape[0] - 1) * self._border_width

        self._output_shape = (new_height, new_width, self._shape[6])
    
    def set(self, images):
        images = np.reshape(np.array(images), self._shape)

        images_list = [None] * (self._shape[3] * 2 - 1)
        images_list[1::2] = [self._sub_border_v] * (self._shape[3] - 1)
        images_list[::2] = [images[:, :, :, i, :, :, :] for i in range(self._shape[3])]
        images = np.concatenate(images_list, axis=4)

        images_list = [None] * (self._shape[1] * 2 - 1)
        images_list[1::2] = [self._border_v] * (self._shape[1] - 1)
        images_list[::2] = [images[:, i, :, :, :, :] for i in range(self._shape[1])]
        images = np.concatenate(images_list, axis=3)

        images_list = [None] * (self._shape[2] * 2 - 1)
        images_list[1::2] = [self._sub_border_h] * (self._shape[2] - 1)
        images_list[::2] = [images[:, i, :, :, :] for i in range(self._shape[2])]
        images = np.concatenate(images_list, axis=1)

        images_list = [None] * (self._shape[0] * 2 - 1)
        images_list[1::2] = [self._border_h] * (self._shape[0] - 1)
        images_list[::2] = [images[i, :, :, :] for i in range(self._shape[0])]
        images = np.concatenate(images_list, axis=0)

        self._image = images
    
    @property
    def shape(self):
        return self._output_shape
    
    @property
    def image(self):
        return self._image


class GifWriter:
    def __init__(self,
                 filename,
                 fps=10,
                 fuzz='5%',
                 ):

        self._filename = filename
        self._fuzz = fuzz

        if self._filename is not None:
            if not os.path.isdir(os.path.dirname(self._filename)):
                os.makedirs(os.path.dirname(self._filename))

            self._writer = imageio.get_writer(self._filename, mode='I', fps=fps)
        else:
            self._writer = None

    def add(self, image):
        self._writer.append_data(image)

    def finish(self):
        if self._writer is not None:
            del self._writer
            self._writer = None

            cmd = ['convert', self._filename, '-fuzz', self._fuzz, '-layers', 'Optimize', self._filename + '.tmp']
            subprocess.check_call(cmd)
            os.remove(self._filename)
            os.rename(self._filename + '.tmp', self._filename)

    def __del__(self):
        self.finish()

def fig_to_numpy(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image
