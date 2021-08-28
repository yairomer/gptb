import os
import subprocess
import time

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
import visdom

def start_visdom(*args, port=9960, **kwargs):
    try:
        vis = visdom.Visdom(port=port, raise_exceptions=True, *args, **kwargs)
    except:
        print('\nRunning visdom at port {}\n'.format(port))
        cmd = ['visdom',
               '-p', str(port),
               ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(3)
        vis = visdom.Visdom(port=port, *args, **kwargs)

    print('Connected to visdom at port {}\n'.format(port))
    return vis

def gen_multigrid_indices(data_shape, borders=None, scale=1):
    if len(data_shape) % 2 == 0:
        data_shape = tuple(data_shape) + (1, )
    data_shape = np.array(data_shape)
    n_dims = len(data_shape)
    n_levels = (n_dims - 1) // 2
    height = data_shape[-3]
    width = data_shape[-2]
    channels = data_shape[-1]

    # data_shape_no_scale = data_shape.copy()
    # data_shape_scaled[-2] *= scale
    # data_shape_scaled[-3] *= scale
    # n_levels = len(data_shape_scaled) // 2

    if borders is None:
        borders = np.arange(n_levels - 1)[::-1]

    def add_border_and_tile(img, reps, border_width):
        img = np.concatenate((img, np.zeros((img.shape[0], border_width), dtype=bool)), axis=1)
        img = np.concatenate((img, np.zeros((border_width, img.shape[1]), dtype=bool)), axis=0)  # pylint: disable=unsubscriptable-object
        img = np.tile(img, reps)
        img = img[:(img.shape[0] - border_width), :(img.shape[1] - border_width)]  # pylint: disable=unsubscriptable-object
        return img

    img = np.ones((height * scale, width * scale), dtype=bool)
    for i_level in range(1, n_levels):
        border_width = borders[-i_level]
        level_height = data_shape[-(i_level * 2 + 3)]
        level_width = data_shape[-(i_level * 2 + 2)]
        img = add_border_and_tile(img, (level_height, level_width), border_width)

    indices = np.repeat(img[:, :, None], channels, axis=2)

    indices_img = np.reshape(np.arange(np.prod(data_shape)), data_shape)
    indices_img = np.repeat(np.repeat(indices_img, scale, axis=n_dims - 2), scale, axis=n_dims - 3)
    shuffle_indices = indices_img.transpose(tuple(np.arange(n_levels) * 2) + tuple(np.arange(n_levels) * 2 + 1) + (n_levels * 2,))
    shuffle_indices = np.reshape(shuffle_indices, (np.prod(shuffle_indices.shape[:n_levels]),np.prod(shuffle_indices.shape[n_levels:(2 * n_levels)]), channels))

    return indices, shuffle_indices

class Images2MultiGrid:
    def __init__(self, data_shape, borders=None, scale=1, bgcolor=None):

        self._indices, self._shuffle_indices = gen_multigrid_indices(data_shape, borders, scale)

        self._shape = self._indices.shape
        self._indices = self._indices.flatten()
        self._shuffle_indices = self._shuffle_indices.flatten()

        if bgcolor is not None:
            self._img = np.ones(self._shape) * np.array(bgcolor)[None, None]
        else:
            self._img = np.zeros(self._shape)
        self(np.repeat(np.linspace(0, 1, np.prod(data_shape[:-1])), data_shape[-1]))

    def __call__(self, imgs):
        self._img.flat[self._indices] = imgs.flat[self._shuffle_indices]  # pylint: disable=unsupported-assignment-operation
        return self._img

    @property
    def shape(self):
        return self._shape

    @property
    def last_image(self):
        return self._img

class Tensor2MultiGrid:
    def __init__(self, data_shape, borders=None, scale=1, device_id='cpu', convert_to_numpy=False):

        self._device_id = device_id
        self._convert_to_numpy = convert_to_numpy

        data_shape_np = data_shape[:-3] + data_shape[-2:] + data_shape[-3:-2]
        self._indices, self._shuffle_indices = gen_multigrid_indices(data_shape_np, borders, scale)

        tensor_indices = np.reshape(np.arange(np.prod(data_shape)), data_shape)
        tensor_indices = np.moveaxis(tensor_indices, -3, -1)
        self._shuffle_indices = tensor_indices.flat[self._shuffle_indices]

        self._indices = np.moveaxis(self._indices, -1, -3)
        self._shuffle_indices = np.moveaxis(self._shuffle_indices, -1, -3)

        self._shape = self._indices.shape
        self._indices = self._indices.flatten()
        self._shuffle_indices = self._shuffle_indices.flatten()

        self._img = torch.zeros(self._shape, device=self._device_id)
        self(torch.repeat_interleave(torch.tensor(np.linspace(0, 1, np.prod(data_shape_np[:-1]), dtype=np.single), device=device_id).view(data_shape[:-3] + (1,) + data_shape[-2:]), data_shape[-3], dim=-3))  # pylint: disable=not-callable

    def __call__(self, imgs):
        imgs = imgs.to(self._device_id)
        # imgs = np.moveaxis(imgs.detach().cpu().numpy(), -3, -1)
        # self._img.flat[self._indices] = imgs.flat[self._shuffle_indices]  # pylint: disable=unsupported-assignment-operation
        # imgs = torch.tensor(self._img).permute(2, 0, 1)

        self._img.view(-1)[self._indices] = imgs.contiguous().view(-1)[self._shuffle_indices]
        return self.last_image

    @property
    def shape(self):
        return self._shape

    @property
    def last_image(self):
        img = self._img
        if self._convert_to_numpy:
            img = np.moveaxis(img.cpu().numpy(), 0, 2)
        return img

def gen_multi_grid(data, borders=None, scale=1, convert_to_numpy=False):
    if isinstance(data, torch.Tensor):
        img = Tensor2MultiGrid(data.shape, borders, scale=scale, device_id=data.device, convert_to_numpy=convert_to_numpy)(data)
    else:
        img = Images2MultiGrid(data.shape, borders, scale=scale)(data)
    return img

class MultiGridImagesViewer:
    def __init__(self, data_shape=None, borders=None, scale=1, use_tensors=None, data=None):
        if data is not None:
            use_tensors = isinstance(data, torch.Tensor)
            data_shape = data.shape
        else:
            if use_tensors is None:
                use_tensors = False
            if data_shape is None:
                raise Exception('data_shape cannot be None if no data is provided')

        if use_tensors:
            self._grid_builder = Tensor2MultiGrid(data_shape, borders, convert_to_numpy=True)
            width = self._grid_builder.shape[2]
            height = self._grid_builder.shape[1]
            channels = self._grid_builder.shape[0]
        else:
            self._grid_builder = Images2MultiGrid(data_shape, borders)
            width = self._grid_builder.shape[1]
            height = self._grid_builder.shape[0]
            channels = self._grid_builder.shape[2]
        if data is not None:
            self._grid_builder(data)

        mpl_dpi = mpl.rcParams['figure.dpi']
        self.fig = plt.figure(figsize=(width / mpl_dpi * scale,
                                       height / mpl_dpi * scale))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.axis('off')
        img = self._grid_builder.last_image
        if channels == 1:
            self._img_image = self.ax.imshow(img[..., 0], cmap='gray')
        else:
            self._img_image = self.ax.imshow(img)
        self.fig.canvas.draw()

    def __call__(self, data):
        img = self._grid_builder(data)
        if img.shape[-1] == 1:
            img = img[..., 0]
        self._img_image.set_data(img)
        self.fig.canvas.draw()

    @property
    def last_image(self):
        return self._grid_builder.last_image


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


class SlicedViz:
    def __init__(self, net, typical_sigma, sliced_range, ref_img, sliced_zoom_range):
        self._net = net

        self._fig, (self._sliced_ax, self._sliced_zoom_ax) = plt.subplots(1, 2, figsize=(8, 4))

        ## Sliced energy
        ## -------------
        self._sliced_range = sliced_range
        self._sliced_zoom_range = sliced_zoom_range
        dir_img = torch.randn_like(ref_img) * typical_sigma
        self._sliced_values = np.sort(np.concatenate([
            np.linspace(-self._sliced_zoom_range, self._sliced_zoom_range, 51, dtype=np.single),
            np.linspace(-self._sliced_range, self._sliced_range, 50, dtype=np.single),
            ], axis=0))
        self._sliced_imgs = ref_img + torch.from_numpy(self._sliced_values).to(ref_img.device)[:, None, None, None] * dir_img[None, :]

        self._sliced_ax.set_title('Sliced energy')
        self._sliced_ax.grid(True)
        self._sliced_line = self._sliced_ax.plot(self._sliced_values, np.zeros_like(self._sliced_values))[0]
        self._sliced_ax.plot(self._sliced_values, self._sliced_values ** 2 / 2, '--k')
        self._sliced_ax.set_xlim(-self._sliced_range, self._sliced_range)
        max_val = self._sliced_range ** 2
        self._sliced_ax.set_ylim(-max_val / 10, max_val)

        self._sliced_zoom_ax.set_title('Sliced energy')
        self._sliced_zoom_ax.grid(True)
        self._sliced_zoom_line = self._sliced_zoom_ax.plot(self._sliced_values, np.zeros_like(self._sliced_values))[0]
        self._sliced_zoom_ax.plot(self._sliced_values, self._sliced_values ** 2 / 2, '--k')
        self._sliced_zoom_ax.set_xlim(-self._sliced_zoom_range, self._sliced_zoom_range)
        self._sliced_zoom_ax.set_ylim(-self._sliced_zoom_range ** 2 / 20, self._sliced_zoom_range ** 2 / 2)

        self.update()

    @property
    def fig(self):
        return self._fig

    def update(self):
        device_id = next(self._net.parameters()).device
        sliced_imgs = self._sliced_imgs.to(device_id)

        with torch.no_grad():
            sliced_energy = self._net(sliced_imgs).view(sliced_imgs.shape[0], -1).mean(dim=1).cpu().numpy() / np.prod(self._sliced_imgs.shape[1:])
        sliced_energy -= sliced_energy[sliced_energy.shape[0] // 2]

        self._sliced_line.set_ydata(sliced_energy)
        max_val = min(self._sliced_range ** 2 / 2, sliced_energy.max())
        self._sliced_ax.set_ylim(-max_val / 10, max_val)
        self._sliced_zoom_line.set_ydata(sliced_energy)
        self._fig.canvas.draw()
