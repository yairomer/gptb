"""
Based on:

- Inception score official: https://github.com/openai/improved-gan/blob/master/inception_score/model.py
- Inception score for pytorch: https://github.com/sbarratt/inception-score-pytorch
- Inception V3 pytorch's implementation: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
- FID official: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
- FID for pytorch (Inception network ported from tf): https://github.com/mseitzer/pytorch-fid
- FID precomputed statistics from: http://bioinf.jku.at/research/ttur/
"""


import os
import argparse
import glob
import shutil
import urllib
import tempfile
import hashlib

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
import PIL

class ISCalculator:
    def __init__(self, network_filename='/tmp/inception_v3.pth', fid_reference_filename='/tmp/fid.npz', download=True):
        if download and not os.path.exists(network_filename):
            network_url = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
            download_url_to_file(network_url, network_filename)

        if download and not os.path.exists(fid_reference_filename):
            fid_reference_url = 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz'  # For CIFAR10
            download_url_to_file(fid_reference_url, fid_reference_filename)

        self._inception_model = InceptionV3(network_filename)
        # self._inception_model = torchvision.models.inception.inception_v3(pretrained=True, transform_input=False)
        # self._inception_upsampler = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)

        reference_data = np.load(fid_reference_filename)
        self._fid_reference_mu = reference_data['mu']
        self._fid_reference_sigma = reference_data['sigma']

    def __call__(self, batch_generator, slices=10, device_id='cpu', n_batches=None):
        inception_model = self._inception_model.to(device_id)
        inception_model.eval()

        predictions = []
        features = []
        with torch.no_grad():
            for imgs in tqdm.tqdm(batch_generator, 'Calculating Inception score', ncols=100, leave=False, total=n_batches):
                # imgs = self._inception_upsampler(imgs.to(device_id))
                predictions_batch, features_batch = inception_model(imgs.to(device_id), intermetiate_features_index=4)

                predictions_batch = torch.nn.functional.softmax(predictions_batch, dim=1).cpu()
                predictions.append(predictions_batch)

                # # If model output is not scalar, apply global spatial average pooling.
                # # This happens if you choose a dimensionality not equal 2048.
                # if features.size(2) != 1 or features.size(3) != 1:
                #     features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
                features_batch = features_batch[:, :, 0, 0].cpu()
                features.append(features_batch)

        predictions = torch.cat(predictions, dim=0).numpy()
        features = torch.cat(features, dim=0).numpy()

        ## Inception score
        ## ---------------
        predictions = predictions.reshape(slices, -1, predictions.shape[1])
        classes_dist = predictions.mean(axis=1, keepdims=True)
        kl_terms = (predictions * (np.log(predictions) - np.log(classes_dist))).sum(axis=2)
        inception_scores = np.exp(kl_terms.mean(axis=1))

        ## FID
        ## ---
        eps = 1e-6
        mu1 = np.mean(features, axis=0)
        sigma1 = np.cov(features, rowvar=False)

        mu2 = self._fid_reference_mu
        sigma2 = self._fid_reference_sigma

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid_score = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

        return inception_scores.mean(), inception_scores.std(), fid_score



class InceptionV3(nn.Module):
    # """Pretrained InceptionV3 network returning feature maps"""

    # # Index of default block of inception to return,
    # # corresponds to output of final average pooling
    # DEFAULT_BLOCK_INDEX = 3

    # # Maps feature dimensionality to their output blocks indices
    # BLOCK_INDEX_BY_DIM = {
    #     64: 0,   # First max pooling features
    #     192: 1,  # Second max pooling featurs
    #     768: 2,  # Pre-aux classifier features
    #     2048: 3  # Final average pooling features
    # }

    def __init__(self,
                 weights_file,
                 ):

        super().__init__()

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE1(1280)
        self.Mixed_7c = InceptionE2(2048)
        self.fc = nn.Linear(2048, 1008)

        state_dict = torch.load(weights_file)
        self.load_state_dict(state_dict)

    def forward(self, x, intermetiate_features_index=None):  # pylint: disable=arguments-differ
        if x.shape[-1] != 299:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)
        # N x 3 x 299 x 299

        x = 2 * x - 1
        # x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        # x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        # x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        intermetiate_features = None

        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73

        if intermetiate_features_index == 1:
            intermetiate_features = x

        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35

        if intermetiate_features_index == 2:
            intermetiate_features = x

        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        #N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        if intermetiate_features_index == 3:
            intermetiate_features = x

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1

        if intermetiate_features_index == 4:
            intermetiate_features = x

        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)

        if intermetiate_features_index is None:
            return x
        else:
            return x, intermetiate_features


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):  # pylint: disable=arguments-differ
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):  # pylint: disable=arguments-differ
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE1(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE1, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE2(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE2, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def run_on_folder(
        path,
        batch_size=100,
        network_filename='/tmp/inception_v3.pth',
        fid_reference_filename='/tmp/fid.npz',
        download=True,
        slices=10,
        device_id='cuda:0',
        ):

    is_calculator = ISCalculator(
        network_filename=network_filename,
        fid_reference_filename=fid_reference_filename,
        download=download,
        )

    files = glob.glob(path)
    n_images = len(files)

    img_pil = PIL.Image.open(files[0])
    img = torchvision.transforms.functional.to_tensor(img_pil)
    batch = torch.empty((batch_size,) + img.shape[1:])

    def batch_generator():
        for i_file in range(n_images):
            img_pil = PIL.Image.open(files[i_file])
            batch[i_file % batch_size] = torchvision.transforms.functional.to_tensor(img_pil)
            if (i_file % batch_size) == (batch_size - 1):
                yield batch

    return is_calculator(batch_generator(), slices=slices, device_id=device_id, n_batches=n_images // batch_size)


def download_url_to_file(url, filename, hash_prefix=None, progress=True):
    # hash_prefix = re.compile(r'-([a-f0-9]*)\.').search(os.path.basename(urllib.parse.urlparse(url))).group(1)

    file_size = None
    u = urllib.request.urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.
    dst_dir = os.path.dirname(filename)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm.tqdm(total=file_size, disable=not progress, unit='B', ncols=100, leave=False, unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")' .format(hash_prefix, digest))
        shutil.move(f.name, filename)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('network', type=str, default='/tmp/inception_v3.pth')
    parser.add_argument('fid_ref', type=str, default='/tmp/fid.npz')
    parser.add_argument("--dont_download", action="store_false")
    parser.add_argument('slices', type=int, default=10)
    parser.add_argument('device_id', type=str, default='cuda:0')

    args = parser.parse_args()

    inception_mean, inception_std, fid_score = run_on_folder(
        path=args['path'],
        batch_size=args['bs'],
        network_filename=args['network'],
        fid_reference_filename=args['fid_ref'],
        download=args['dont_download'],
        slices=args['slices'],
        device_id=args['device_id'],
        )

    print('Inception score (mean): {}'.format(inception_mean))
    print('Inception score std: {}'.format(inception_std))
    print('FID score: {}'.format(fid_score))
