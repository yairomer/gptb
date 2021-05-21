import os
import sys
import shutil
import random
import time
import urllib
import tempfile
import hashlib
import io
from contextlib import contextmanager

import numpy as np
import torch
import tqdm
from line_profiler import LineProfiler


class Profiler:
    def __init__(self, enable=True, output_filename='./profile.txt'):
        self._enable = enable
        self._funcs_to_profile = []
        self._output_filenmae = output_filename

    @property
    def enable(self):
        return self._enable

    def add_function(self, func):
        if self.enable:
            self._funcs_to_profile.append(func)
        return func

    @contextmanager
    def run_and_profile(self):
        if len(self._funcs_to_profile) > 0:
            profiler = LineProfiler()
            for func in self._funcs_to_profile:
                profiler.add_function(func)
            profiler.enable_by_count()
try:
                yield
            finally:
                with io.StringIO() as str_stream:
                    profiler.print_stats(str_stream)
                    string = str_stream.getvalue()
                print(f'Writing profile data to "{self._output_filenmae}"')
                with open(self._output_filenmae, 'w') as fid:
                    fid.write(string)
        else:
            yield


class CountDowner:
    def __init__(self, interval, reset=False):
        self._interval = interval
        self._last_reset = 0
        if reset:
            self.reset()

    def __bool__(self):
        return time.time() > self._last_reset + self._interval

    def reset(self):
        self._last_reset = time.time()


def set_random_state(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    rand_gen = np.random.RandomState(random_seed)  # pylint: disable=no-member
    return rand_gen

class Resevior:
    def __init__(self, reservior_size, data=tuple(), indices=None, keep_n_first=0, keep_n_last=0, last_proposal=None, rand_seed=0):

        self._reservior_size = reservior_size
        self._keep_n_first = keep_n_first
        self._keep_n_last = keep_n_last
        if isinstance(rand_seed, np.random.RandomState):  # pylint: disable=no-member
            self._rand_gen = rand_seed
        else:
            self._rand_gen = np.random.RandomState(rand_seed)  # pylint: disable=no-member

        self._data = []
        self._indices = []
        self._last_proposal = None

        if len(data) > 0:
            if indices is None:
                for val in data:
                    self.add(val)
            else:
                for index, val in zip (data, indices):
                    self.add(index, val)

        if not last_proposal is None:
            self._last_proposal = last_proposal

    @property
    def last(self):
        if len(self._indices) == 0:
            return 0
        else:
            if self._keep_n_last > 0:
                return self._indices[-1]
            else:
                return self._last_proposal

    def add(self, index, val=None):
        if val is None:
            val = index
            index = self.last + 1

        term_to_remove = None 
        index_to_remove = None

        self._data.append(val)
        self._indices.append(index)

        if len(self._indices) > self._reservior_size:
            effective_size = self._reservior_size - self._keep_n_first -  self._keep_n_last
            candidate_range = self._indices[-self._keep_n_last - 1] - self._last_proposal
            effective_range = self._indices[-self._keep_n_last - 1] - (0 if self._keep_n_first == 0 else self._indices[self._keep_n_first - 1])
            # chance_to_accept = 1 - (1 - candidate_range / effective_range) ** effective_size
            chance_to_accept = candidate_range / effective_range * effective_size

            if self._rand_gen.rand() < chance_to_accept:
                term_to_remove = self._rand_gen.choice(effective_size) + self._keep_n_first
            else:
                term_to_remove = len(self._indices) - 1 - self._keep_n_last

            self._last_proposal = self._indices[-self._keep_n_last - 1]
            self._data.pop(term_to_remove)
            index_to_remove = self._indices.pop(term_to_remove)
        else:
            if len(self._indices) > self._keep_n_last:
                self._last_proposal = self._indices[-self._keep_n_last - 1]
            else:
                self._last_proposal = self._indices[0]

        return term_to_remove, index_to_remove

    def __len__(self):
        return len(self._indices)

    def state_dict(self):
        state = {
            'reservior_size': self._reservior_size,
            'data': self._data,
            'indices': self._indices,
            'keep_n_first': self._keep_n_first,
            'keep_n_last': self._keep_n_last,
            'last_proposal': self._last_proposal,
            }

        return state

# def download_file(url, filename):
#     if not os.path.exists(os.path.dirname(filename)):
#         os.makedirs(os.path.dirname(filename))
#     def _progress(count, block_size, total_size):
#         sys.stdout.write('\r>> Downloading network %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
#         sys.stdout.flush()
#     compressed_netowrk_filename, _ = urllib.request.urlretrieve(url, filename, _progress)
#     print()
#     statinfo = os.stat(compressed_netowrk_filename)
#     print('Succesfully downloaded network', statinfo.st_size, 'bytes.')

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
