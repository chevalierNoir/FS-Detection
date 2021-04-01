import torch
import numpy as np
import torch.utils.data as tud
from collections import defaultdict

if torch.__version__ == '0.4.0':
    Sampler = tud.sampler.Sampler
else:
    Sampler = tud.Sampler

class BucketBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    Example:
        >>> list(BucketBatchSampler(shuffle=True, batch_size=2, files=['f0']*3+['f1']*5))
        [[7, 5], [3, 4], [6], [1, 2], [0]]
    """
    def __init__(self, shuffle, batch_size, files, cycle=True, seeds=list(range(100))):
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle should be a boolean value,"
                             "but got shuffle={}"
                             .format(shuffle))
        if not (isinstance(batch_size, int) or isinstance(batch_size, dict)):
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(files, list):
            raise ValueError("files should be a list type, but got "
                             "files={}".format(files))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files = files
        fidx = defaultdict(list)
        for ix, f in enumerate(files):
            fidx[f].append(ix)
        self.fidx = list(fidx.values())
        self.cycle = cycle
        print("Initialize seeds and index array")
        seed = seeds.pop()
        batch_indexes = self.make_batches(self.fidx, seed, cycle, shuffle, batch_size)
        self.state_dict_ = {'seeds': seeds, 'batch_indexes': batch_indexes}

    def __iter__(self):
        if len(self.state_dict_['batch_indexes']) == 0:
            seed = self.state_dict_['seeds'].pop() if len(self.state_dict_['seeds']) > 0 else 222
            self.state_dict_['batch_indexes'] = self.make_batches(self.fidx, seed, self.cycle, self.shuffle, self.batch_size)
        while True:
            if len(self.state_dict_['batch_indexes']) > 0:
                batch = self.state_dict_['batch_indexes'].pop()
                yield batch
            else:
                break

    def __len__(self):
        return len(self.state_dict_['batch_indexes'])

    def make_batches(self, file_indexes, seed, cycle, shuffle, batch_size):
        perm = []
        batch_indexes = []
        if shuffle:
            np.random.seed(seed)
            xs = np.random.permutation(len(file_indexes))
            for x in xs:
                perm.append(np.random.permutation(file_indexes[x]).tolist())
        else:
            perm = file_indexes
        for index_group in perm:
            batch = []
            for index in index_group:
                batch.append(index)
                if len(batch) == batch_size:
                    batch_indexes.append(batch)
                    batch = []
            if len(batch) > 0:
                if cycle:
                    batch = batch + index_group[:batch_size - len(batch)]
                batch_indexes.append(batch)
        return batch_indexes

    def load_state_dict(self, state_dict):
        # Note seeds and self.seeds is same obj
        self.state_dict_ = state_dict
        print("Loading sampler state dict, %d batches, %d seeds" % (len(self.state_dict_['batch_indexes']), len(self.state_dict_['seeds'])))
        return

    def state_dict(self):
        return self.state_dict_
