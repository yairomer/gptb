import os
import subprocess

import numpy as np
import imageio

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
