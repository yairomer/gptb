import random
import time

import numpy as np
import torch

try:
    profile_func = profile
except NameError:
    def profile_func(func):
        return func

class CountDowner:
    def __init__(self, interval, reset=False):
        self._interval = interval
        self._last_reset = 0
        if reset:
            self.reset()

    def __bool__(self):
        return time.time() >  self._last_reset + self._interval

    def reset(self):
        self._last_reset = time.time()


def set_random_state(random_seed):
    random.seed(random_seed)
    # torch.backends.cudnn.deterministic = True  # !!Warning!!: This makes things run A LOT slower
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    rand_gen = np.random.RandomState(random_seed)
    return rand_gen

class Resevior:
    def __init__(self, reservior_size, data=[], indices=None, keep_n_first=0, keep_n_last=0, last_proposal=None, rand_seed=0):

        self._reservior_size = reservior_size
        self._keep_n_first = keep_n_first
        self._keep_n_last = keep_n_last
        if isinstance(rand_seed, np.random.RandomState):
            self._rand_gen = rand_seed
        else:
            self._rand_gen = np.random.RandomState(rand_seed)

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

            if self._rand_gen.rand() <  chance_to_accept:
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
