import os
import struct
import glob
import warnings

import torch
import torchvision
import PIL

import numpy as np
from torchvision import transforms

def load_dataset(dataset_name, *args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == 'spiral':
        dataset = load_spiral(*args, **kwargs)
    elif dataset_name == 'mnist':
        dataset = load_mnist(*args, **kwargs)
    elif dataset_name == 'celeba':
        dataset = load_celeba(*args, **kwargs)
    elif dataset_name == 'cifar10':
        dataset = load_cifar10(*args, **kwargs)
    elif dataset_name == 'fixed_imagenet':
        dataset = load_fixed_image_net(*args, **kwargs)
    elif dataset_name == 'files':
        dataset = load_files(*args, **kwargs)
    else:
        raise Exception('Unknown dataset: "{}"'.format(dataset_name))
    return dataset


class DatasetWrapper:
    def __init__(self,
                 dataset,
                 drop_labels=False,
                 store_used_before=False,
                 store_used_after=False,
                 asumme_fixed_size=False,
                 transform=None,
                 to_device=None,
                 ):

        self._dataset = dataset
        self._transform = transform
        self._drop_labels = drop_labels
        self._store_used_before = store_used_before
        self._store_used_after = store_used_after
        self._to_device = to_device

        if self._store_used_before or self._store_used_after:
            self._was_used = np.zeros(len(self._dataset), dtype=bool)
            self._used_labels = [None] * len(self._dataset)
        else:
            self._was_used = None
            self._used_labels = None

        if self._store_used_before:
            if asumme_fixed_size:
                data = self._dataset[0][0]
                self._used_data = torch.zeros((len(self._dataset), ) + data.shape, dtype=data.dtype, device=self._to_device or 'cpu')
            else:
                self._used_data = [None] * len(self._dataset)
        elif self._store_used_after:
            data = self._dataset[0][0]
            if self._transform is not None:
                data = self._transform(data)
            self._used_data = torch.zeros((len(self._dataset), ) + data.shape, dtype=data.dtype, device=self._to_device or 'cpu')
        else:
            self._used_data = None

    def __getitem__(self, index):
        if self._store_used_after and self._was_used[index]:
            data = self._used_data[index]
            label = self._used_labels[index]
        else:
            if self._store_used_before and self._was_used[index]:
                data = self._used_data[index]
                label = self._used_labels[index]
            else:
                data, label = self._dataset[index]
                if self._store_used_before:
                    self._used_data[index] = data
                    self._used_labels[index] = label
                    self._was_used[index] = True

            if self._transform is not None:
                data = self._transform(data)

            if self._to_device is not None:
                data = data.to(self._to_device)

            if self._store_used_after:
                self._used_data[index] = data
                self._used_labels[index] = label
                self._was_used[index] = True

        if self._drop_labels:
            return data
        else:
            return data, label

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, attr):
        if hasattr(self._dataset, attr):
            return getattr(self._dataset, attr)
        else:
            raise AttributeError("{} object has no attribute {}".format(self._dataset.__class__.__name__, attr))


def load_spiral(n_rounds=2,
                scale=1,
                noise_std=0.01,
                n_samples=1024,
                rand_seed=0,
                ):

    if isinstance(rand_seed, np.random.RandomState):  # pylint: disable=no-member
        rand_gen = rand_seed
    else:
        rand_gen = np.random.RandomState(rand_seed)  # pylint: disable=no-member

    t = rand_gen.rand(n_samples)
    r = t * scale
    theta = t * n_rounds * 2 * np.pi
    data = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1)
    data += rand_gen.randn(n_samples, 2) * noise_std
    dataset = torch.tensor(data, dtype=torch.float)  # pylint: disable=not-callable
    return dataset

def generate_spiral_line(n_rounds=2,
                         scale=1,
                         n_points=1000,
                         ):

    t = np.linspace(0, 1, n_points)
    r = t * scale
    theta = t * n_rounds * 2 * np.pi
    data = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1)
    return data


def load_mnist(data_folder,
               split='train',
               n_samples_val=256,
               drop_labels=False,
               additional_transform=None,
               rand_seed=0,
               ):

    rand_gen = np.random.RandomState(rand_seed)  # pylint: disable=no-member

    if split in ['train', 'val']:
        images_filename = os.path.join(data_folder, 'train-images-idx3-ubyte')
        labels_filename = os.path.join(data_folder, 'train-labels-idx1-ubyte')
    else:
        images_filename = os.path.join(data_folder, 't10k-images-idx3-ubyte')
        labels_filename = os.path.join(data_folder, 't10k-labels-idx1-ubyte')

    with open(images_filename, 'rb') as fid:
        _, _, rows, cols = struct.unpack(">IIII", fid.read(16))
        images = np.fromfile(fid, dtype=np.uint8).reshape(-1, rows, cols)

    with open(labels_filename, 'rb') as fid:
        _, _ = struct.unpack(">II", fid.read(8))
        labels = np.fromfile(fid, dtype=np.int8)

    images = images / 255.

    dataset = torch.utils.data.TensorDataset(torch.tensor(images[:, None, :, :], dtype=torch.float),  # pylint: disable=not-callable
                                             torch.tensor(labels, dtype=torch.int),  # pylint: disable=not-callable
                                             )

    if split == 'train':
        indices = rand_gen.permutation(len(dataset))[:-n_samples_val]
        dataset = torch.utils.data.Subset(dataset, indices)
    elif split == 'val':
        indices = rand_gen.permutation(len(dataset))[-n_samples_val:]
        dataset = torch.utils.data.Subset(dataset, indices)

    if drop_labels or (additional_transform is not None):
        if additional_transform is not None:
            additional_transform = torchvision.transforms.Compose(additional_transform)
        dataset = DatasetWrapper(dataset, drop_labels=drop_labels, transform=additional_transform)

    return dataset

def load_celeba(data_folder,
                split='train',
                crop_size=178,
                resize_to=64,
                target_type=['attr', 'identity', 'bbox', 'landmarks'],
                use_grayscale=False,
                drop_labels=False,
                store_used=True,
                normalize=False,
                additional_transform=None,
                download=False,
                to_device=None,
                ):

    ## torchvision adds the celeba folder to the path
    data_folder = os.path.abspath(data_folder)
    assert os.path.basename(data_folder) == 'celeba', 'data_folder name must end with "celeba"'
    data_folder = os.path.abspath(os.path.join(data_folder, '../'))

    transform = [
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.Resize(resize_to),
        ]
    if use_grayscale:
        transform += [torchvision.transforms.Grayscale()]
    if additional_transform is not None:
        transform += additional_transform
    transform += [torchvision.transforms.ToTensor()]
    if normalize:
        transform += [torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = torchvision.transforms.Compose(transform)

    dataset = torchvision.datasets.CelebA(data_folder, split=split, target_type=target_type, transform=transform, download=download)

    dataset = DatasetWrapper(dataset, drop_labels=drop_labels, store_used_after=store_used, to_device=to_device)

    return dataset

def get_celeba_attr_list(data_folder):
    with open(os.path.join(data_folder, 'list_attr_celeba.txt'), 'r') as fid:
        fid.readline()
        attr_list = fid.readline().split()
    return attr_list

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_cifar10(data_folder,
                 split='train',
                 drop_labels=False,
                 store_used=True,
                 use_grayscale=False,
                 n_samples_train=None,
                 n_samples_val=256,
                 normalize=False,
                 additional_transform=None,
                 download=False,
                 to_device=None,
                 rand_seed=0,
                 ):

    rand_gen = np.random.RandomState(rand_seed)  # pylint: disable=no-member

    transform = []
    if use_grayscale:
        transform += [torchvision.transforms.Grayscale()]
    if additional_transform is not None:
        transform += additional_transform
    transform += [torchvision.transforms.ToTensor()]
    if normalize:
        transform += [transforms.Normalize(cifar10_mean, cifar10_std)]
    transform = torchvision.transforms.Compose(transform)

    dataset = torchvision.datasets.CIFAR10(data_folder, train=(split in ('train', 'val')), transform=transform, download=download)

    if (n_samples_val is not None) or (n_samples_train is not None):
        if n_samples_train is None:
            n_samples_train = len(dataset) - n_samples_val

        indices = rand_gen.permutation(len(dataset))
        if split == 'train':
            dataset = torch.utils.data.Subset(dataset, indices[:n_samples_train])
        elif split == 'val':
            dataset = torch.utils.data.Subset(dataset, indices[-n_samples_val:])

    dataset = DatasetWrapper(dataset, drop_labels=drop_labels, store_used_after=store_used, to_device=to_device)

    return dataset

def get_cifar10_denormalize():
    return transforms.Normalize(
        tuple(-m/s for m, s in zip(cifar10_mean, cifar10_std)),
        tuple(1/s for s in cifar10_std),
        )


def load_fixed_image_net(data_folder,
                         split='train',
                         resize=True,
                         image_size=256,
                         drop_labels=False,
                         store_used=True,
                         use_grayscale=False,
                         rand_seed=0,
                         n_samples_train=None,
                         n_samples_val=256,
                         normalize=False,
                         additional_transform=None,
                         ):

    rand_gen = np.random.RandomState(rand_seed)  # pylint: disable=no-member

    transform = []
    if resize:
        transform += [torchvision.transforms.Resize(image_size)]
        transform += [torchvision.transforms.CenterCrop(image_size)]
    else:
        transform += [torchvision.transforms.RandomCrop(image_size, pad_if_needed=True, padding_mode='edge')]
    if use_grayscale:
        transform += [torchvision.transforms.Grayscale()]
    if additional_transform is not None:
        transform += additional_transform
    transform += [torchvision.transforms.ToTensor()]
    if normalize:
        transform += [torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = torchvision.transforms.Compose(transform)

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_folder, 'train' if split in ('train', 'val') else 'val'),
        transform=transform,
        )

    if (n_samples_val is not None) or (n_samples_train is not None):
        if n_samples_train is None:
            n_samples_train = len(dataset) - n_samples_val

        indices = rand_gen.permutation(len(dataset))
        if split == 'train':
            dataset = torch.utils.data.Subset(dataset, indices[:n_samples_train])
        elif split == 'val':
            dataset = torch.utils.data.Subset(dataset, indices[-n_samples_val:])

    dataset = DatasetWrapper(dataset, drop_labels=drop_labels, store_used_after=store_used)

    return dataset


def load_files(files,
               split=None,
               n_samples_val=256,
               n_samples_test=256,
               minimal_width=None,
               minimal_height=None,
               resize_to=None,
               additional_transform=None,
               store_used=False,
               to_device=None,
               rand_seed=0,
               ):


    transform = []
    if resize_to is not None:
        transform += [torchvision.transforms.Resize(resize_to)]
    if additional_transform is not None:
        transform += additional_transform
    dataset = Files(files,
                    minimal_width=minimal_width,
                    minimal_height=minimal_height,
                    additional_transform=transform,
                    fake_label=0,
                    )


    rand_gen = np.random.RandomState(rand_seed)  # pylint: disable=no-member
    indices = np.arange(len(dataset))
    rand_gen.shuffle(indices)
    indices_train = indices[:-(n_samples_val + n_samples_test)]
    indices_val = indices[-(n_samples_val + n_samples_test):-n_samples_test]
    indices_test = indices[-n_samples_test:]
    if split == 'train':
        dataset = torch.utils.data.Subset(dataset, indices_train)
    elif split == 'val':
        dataset = torch.utils.data.Subset(dataset, indices_val)
    elif split == 'test':
        dataset = torch.utils.data.Subset(dataset, indices_test)

    dataset = DatasetWrapper(dataset, drop_labels=True, store_used_after=store_used, to_device=to_device)

    return dataset


class Files:
    def __init__(self, files, minimal_width=None, minimal_height=None, additional_transform=None, fake_label=None):
        self._fake_label = fake_label
        self._files = glob.glob(files)
        self._files.sort()
        if (minimal_width is not None) or (minimal_height is not None):
            if minimal_width is None:
                minimal_width = 0
            if minimal_height is None:
                minimal_height = 0
            files = []
            for img_file in self._files:
                img_shape = PIL.Image.open(img_file).size
                if (img_shape[1] >= minimal_height) and (img_shape[0] >= minimal_width):
                    files.append(img_file)
            self._files = files

        transform = []
        if additional_transform is not None:
            transform += additional_transform
        transform += [torchvision.transforms.ToTensor()]
        self._transform = torchvision.transforms.Compose(transform)

    def __getitem__(self, index):
        img_pil = PIL.Image.open(self._files[index])
        img = self._transform(img_pil)
        if self._fake_label is not None:
            return img, self._fake_label
        else:
            return img

    def __len__(self):
        return len(self._files)
