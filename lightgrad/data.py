import numpy as np
from math import ceil
from typing import Tuple
from .utils import fetch
from .autograd import Tensor

class Dataset(object):
    def __init__(self, tensors:Tuple[Tensor], shuffle:bool =True, batchsize:int =8) -> None:
        assert all((t.shape[0] == tensors[0].shape[0] for t in tensors[1:]))
        self.__tensors = tensors
        self.__shuffle, self.__bs = shuffle, batchsize
    @property
    def n(self) -> int:
        return self.__tensors[0].shape[0]
    def shuffle(self) -> None:
        shuffle_idx = np.random.permutation(self.n)
        self.__tensors = tuple(t[shuffle_idx].detach() for t in self.__tensors)

    def __getitem__(self, idx) -> Tuple[Tensor]:
        outs = tuple(t[idx, ...].detach() for t in self.__tensors)
        return outs
    def __iter__(self) -> iter:
        if self.__shuffle:
            self.shuffle()
        for i in range(len(self)):
            yield self[i*self.__bs : (i+1)*self.__bs]
    def __len__(self) -> int:
        return ceil(self.n / self.__bs)


""" MNIST Dataset """

class MNIST(Dataset):
    def __init__(self, train:bool =True, n:int =60_000, **kwargs):
        import gzip
        n = min(n, 60_000 if train else 10_000)
        # build urls
        images_url = "http://yann.lecun.com/exdb/mnist/%s" % ("train-images-idx3-ubyte.gz" if train else "t10k-images-idx3-ubyte.gz")
        labels_url = "http://yann.lecun.com/exdb/mnist/%s" % ("train-labels-idx1-ubyte.gz" if train else "t10k-labels-idx1-ubyte.gz")
        # fetch and parse
        parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
        X_raw = parse(fetch(images_url))[0x10:0x10 + n*28*28].reshape((-1, 28, 28)).astype(np.float32)
        Y_raw = parse(fetch(labels_url))[8:8+n].astype(np.int16)
        # initialize dataset
        Dataset.__init__(self, (
            Tensor.from_numpy(X_raw / 255, requires_grad=False), 
            Tensor.from_numpy(Y_raw, requires_grad=False)
        ), **kwargs)
