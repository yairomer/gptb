import os
import shutil
import time
import subprocess
import socket
import glob
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import tensorboardX
import tensorflow as tf
import imageio


class NetworkState:
    def __init__(self,
                 tensorboard_folder=None,
                 checkpoints_folder=None,
                 output_folder=None,
                 version='',
                 modules=None,
                 optimizers=None,
                 tensors=None,
                 arrays=None,
                 metadata=None,
                 ):

        ## Define parameters
        ## =================
        self.step = 0

        self._tensorboard_folder = tensorboard_folder
        self._checkpoints_folder = checkpoints_folder
        self._output_folder = output_folder
        self._version = version

        self.tensorboard_logger = None

        if modules is None:
            self._modules = {}
        else:
            self._modules = modules

        if optimizers is None:
            self._optimizers = {}
        else:
            self._optimizers = optimizers

        if tensors is None:
            self._tensors = {}
        else:
            self._tensors = tensors

        if arrays is None:
            self._arrays = {}
        else:
            self._arrays = arrays

        if metadata is None:
            self._metadata = {}
        else:
            self._metadata = metadata

    @property
    def output_folder(self):
        return os.path.join(self._output_folder, self._version)

    def reset_folders(self):
        if self._tensorboard_folder is not None and os.path.isdir(os.path.join(self._tensorboard_folder, self._version)):

            ## Shutdown existing tensorboards
            ps = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE).stdout.readlines()
            headers = ps[0].decode("utf-8").strip().split()
            cmd_index = headers.index('COMMAND')
            pid_index = headers.index('PID')
            for line in ps[1:]:
                line = line.decode("utf-8").strip().split(None, len(headers) - 1)
                cmd = line[cmd_index]
                if ('tensorboard' in cmd) and (self._tensorboard_folder in cmd):
                    print('Killing process: {}'.format(' '.join(line)))
                    pid = int(line[pid_index])
                    os.kill(pid, 9)
                    time.sleep(5)

            shutil.rmtree(os.path.join(self._tensorboard_folder, self._version))

        if self._checkpoints_folder is not None and os.path.isdir(os.path.join(self._checkpoints_folder, self._version)):
            shutil.rmtree(os.path.join(self._checkpoints_folder, self._version))

        if self._output_folder is not None and os.path.isdir(os.path.join(self._output_folder, self._version)):
            shutil.rmtree(os.path.join(self._output_folder, self._version))

    def save_step(self):
        filename = os.path.join(self._checkpoints_folder, self._version, 'periodic', 'checkpoint_{}.pt'.format(self.step))
        self.save(filename)

    def save_best(self):
        filename = os.path.join(self._checkpoints_folder, self._version, 'checkpoint_best.pt')
        self.save(filename)

    def gen_reservior_saver(self, reservior_size, rand_seed=0):
        return ReseviorSaver(self, reservior_size=reservior_size, previous_checkpoint_step=self.step, rand_seed=rand_seed)

    def save(self, filename=None):
        if filename is None:
            filename = os.path.join(self._checkpoints_folder, self._version, 'checkpoint.pt')

        checkpoint = {
            'step': self.step,
            'modules': {},
            'optimizers': {},
            'tensors': {},
            'arrays': {},
            'metadata': self._metadata,
            }
        for module_name, module in self._modules.items():
            checkpoint['modules'][module_name] = module.state_dict()
        for optimizer_name, optimizer in self._optimizers.items():
            checkpoint['optimizers'][optimizer_name] = optimizer.state_dict()
        for tensor_name, tensor in self._tensors.items():
            checkpoint['tensors'][tensor_name] = tensor
        for array_name, array in self._arrays.items():
            checkpoint['arrays'][array_name] = array.tobytes()

        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        torch.save(checkpoint, filename)

    def to_device(self, device_id):
        for module in self._modules.values():
            module.to(device_id)
        # for optimizer in self._optimizers.values:
        #     optimizer.to(device_id)

    def list_checkpoints_steps(self):
        checkpoints_steps = []
        if os.path.isdir(os.path.join(self._checkpoints_folder, self._version, 'periodic')):
            matches = map(lambda x: re.match('^checkpoint_(\\d+)\\.pt$', x),
                        os.listdir(os.path.join(self._checkpoints_folder, self._version, 'periodic')))
            checkpoints_steps = sorted([int(x.groups()[0]) for x in matches if x is not None])
        return checkpoints_steps

    def load_step(self, step=None):
        if step is None:
            checkpoints_steps = self.list_checkpoints_steps()
            if len(checkpoints_steps) > 0:
                setp = max(checkpoints_steps)

        if step is not None:
            filename = os.path.join(self._checkpoints_folder, self._version, 'periodic', 'checkpoint_{}.pt'.format(step))
            self.load(filename)

    def load_best(self, load_to_cpu=False):
        filename = os.path.join(self._checkpoints_folder, self._version, 'checkpoint_best.pt')
        self.load(filename, load_to_cpu=load_to_cpu)

    def load(self, filename=None, load_to_cpu=False):
        if filename is None:
            filename = os.path.join(self._checkpoints_folder, self._version, 'checkpoint.pt')
            if not os.path.isfile(filename):
                return False

        if load_to_cpu:
            checkpoint = torch.load(filename, map_location='cpu')
        else:
            checkpoint = torch.load(filename)

        self.step = checkpoint['step']
        for module_name in self._modules.keys():
            self._modules[module_name].load_state_dict(checkpoint['modules'][module_name])
        for optimizer_name in self._optimizers.keys():
            self._optimizers[optimizer_name].load_state_dict(checkpoint['optimizers'][optimizer_name])
        for tensor_name in checkpoint['tensors'].keys():
            if tensor_name in self._tensors.keys():
                self._tensors[tensor_name][:] = checkpoint['tensors'][tensor_name][:]
            else:
                self._tensors[tensor_name] = checkpoint['tensors'][tensor_name]
        for array_name in self._arrays.keys():
            self._arrays[array_name].frombuffer(checkpoint['arrays'][array_name])
        self._metadata = checkpoint.get('metadata', {})

        return True

    def remove_checkpoint(self, step):
        filename = os.path.join(self._checkpoints_folder, self._version, 'periodic', 'checkpoint_{}.pt'.format(step))
        if os.path.isfile(filename):
            os.remove(filename)

    def increment(self):
        self.step += 1

    def init_tensorboard(self, start_board=False, current_version_only=True):
        if self._tensorboard_folder is not None:
            self.tensorboard_logger = tensorboardX.SummaryWriter(os.path.join(self._tensorboard_folder, self._version))
            if start_board:
                self.run_tensorboard(current_version_only=current_version_only, in_background=True)

    def run_tensorboard(self,
                        current_version_only=True,
                        in_background=False,
                        initial_port=9970,
                        ):
        if current_version_only:
            tensorboard_folder = os.path.join(self._tensorboard_folder, self._version)
        else:
            tensorboard_folder = self._tensorboard_folder

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
        print('\nRunning tensorboard at port {}\n'.format(port))
        cmd = ['tensorboard',
               '--port', str(port),
               '--logdir', tensorboard_folder,
               '--samples_per_plugin', 'images=100',
               '--reload_interval=5',
               ]
        if in_background:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def scalar_from_tensorboard(self, tag):
        tensorboard_filename = glob.glob(os.path.join(self._tensorboard_folder, self._version, 'events.out.tfevents.*'))[0]

        data = []
        steps = []
        sess = tf.InteractiveSession()
        with sess.as_default():
            for msg in tf.train.summary_iterator(tensorboard_filename):
                for value in msg.summary.value:
                    if value.tag == tag:
                        data.append(value.simple_value)
                        steps.append(msg.step)

        data = np.array(data)
        steps = np.array(steps)

        return steps, data

    def images_from_tensorboard(self, tag):
        tensorboard_filename = glob.glob(os.path.join(self._tensorboard_folder, self._version, 'events.out.tfevents.*'))[0]

        image_str = tf.placeholder(tf.string)
        im_tf = tf.image.decode_image(image_str)
        sess = tf.InteractiveSession()
        with sess.as_default():
            for msg in tf.train.summary_iterator(tensorboard_filename):
                for value in msg.summary.value:
                    if value.tag == tag:
                        yield msg.step, im_tf.eval({image_str: value.image.encoded_image_string})

    def gif_from_tensorboard(self, tag, gif_filename=None, step=1, fps=3, repeat_last=0):
        if gif_filename is None:
            gif_filename = os.path.join(self._output_folder, self._version, '{}.gif'.format(tag.replace('/', '_')))

        if not os.path.isdir(os.path.dirname(gif_filename)):
            os.makedirs(os.path.dirname(gif_filename))

        index = 0
        with imageio.get_writer(gif_filename, mode='I', fps=fps) as writer:
            for current_step, image in self.images_from_tensorboard(tag):
                if ((isinstance(step, int) and index % step == 0)) or (not isinstance(step, int) and (current_step in step)):
                    writer.append_data(image)
                    last_image = image
                    print(image.mean())
                    print(current_step)
                index += 1
            for _ in range(repeat_last):
                writer.append_data(last_image)

        cmd = ['convert', gif_filename, '-fuzz', '10%', '-layers', 'Optimize', gif_filename + '.tmp']
        print('Running: "{}"'.format(' '.join(cmd)))
        subprocess.check_call(cmd)
        os.remove(gif_filename)
        os.rename(gif_filename + '.tmp', gif_filename)

class ReseviorSaver:
    def __init__(self, net_state, reservior_size, previous_checkpoint_step, rand_seed=0):
        self._net_state = net_state
        self._reservior_size = reservior_size
        self._previous_checkpoint_step = previous_checkpoint_step
        if isinstance(rand_seed, np.random.RandomState):
            self._rand_gen = rand_seed
        else:
            self._rand_gen = np.random.RandomState(rand_seed)

    def save(self):
        was_saved = False
        checkpoints_steps = self._net_state.list_checkpoints_steps()
        if len(checkpoints_steps) < self._reservior_size:
            self._net_state.save_step()
            was_saved = True
        else:
            if self._rand_gen.rand() <  (1 - self._previous_checkpoint_step / self._net_state.step) / self._reservior_size:
                self._net_state.save_step()
                was_saved = True
                self._net_state.remove_checkpoint(self._rand_gen.choice(checkpoints_steps))

        self._previous_checkpoint_step = self._net_state.step

        return was_saved

class LayerNorm(nn.Module):

    def __init__(self, shape, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self._affine = affine
        self._eps = eps

        if self._affine:
            self._weights = nn.Parameter(torch.ones((1, np.prod(shape))))
            self._bias = nn.Parameter(torch.zeros((1, np.prod(shape))))

    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], -1)

        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        x = (x - mean) / (std + self._eps)

        if self._affine:
            x = self._weights * x + self._bias

        x = x.view(*shape)

        return x


def weight_init(m):
    '''
    Taken from: https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, (nn.Sequential,
                        nn.Dropout,
                        nn.ReLU,
                        nn.LeakyReLU,
                        nn.Sigmoid,
                        nn.Tanh,
                        )):
        pass
    else:
        print("!! Warning: {} has no deafault initialization scheme".format(type(m)))


def images_from_tensorboard(tensorboard_folder, tag, step=-1):
    if isinstance(step, int):
        single_image = True
        step = [step]

    tensorboard_filename = glob.glob(os.path.join(tensorboard_folder, 'events.out.tfevents.*'))[0]

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)
    sess = tf.InteractiveSession()
    last_msg = None
    with sess.as_default():
        images = []
        for msg in tf.train.summary_iterator(tensorboard_filename):
            if msg.step in step:
                for value in msg.summary.value:
                    if value.tag == tag:
                        images.append(im_tf.eval({image_str: value.image.encoded_image_string}))
            last_msg = msg

        if len(step) == 1 and step[0] == -1:
            for value in last_msg.summary.value:
                if value.tag == tag:
                    images.append(im_tf.eval({image_str: value.image.encoded_image_string}))

    if single_image:
        images = images[0]
    else:
        images = np.array(images)

    return images


def gif_from_tensorboard(gif_filename, tensorboard_folder, tag, step=1, fps=3, repeat_last=0):
    tensorboard_filename = glob.glob(os.path.join(tensorboard_folder, 'events.out.tfevents.*'))[0]

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)
    sess = tf.InteractiveSession()
    index = 0
    with sess.as_default():
        with imageio.get_writer(gif_filename, mode='I', fps=fps) as writer:
            for msg in tf.train.summary_iterator(tensorboard_filename):
                for value in msg.summary.value:
                    if value.tag == tag:
                        if ((isinstance(step, int) and index % step == 0)) or (not isinstance(step, int) and (msg.step in step)):
                            image = im_tf.eval({image_str: value.image.encoded_image_string})
                            writer.append_data(image)
                        index += 1
            for _ in range(repeat_last):
                writer.append_data(image)


def add_indicies(dataset_class):
    class IndexedDataset(dataset_class):
        def __getitem__(self, index):
            x = super().__getitem__(index)
            if isinstance(x, tuple):
                return (index,) + x
            else:
                return index, x

    IndexedDataset.__name__ = 'Indexed' + dataset_class.__name__

    return IndexedDataset
