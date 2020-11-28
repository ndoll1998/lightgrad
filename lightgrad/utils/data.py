import numpy as np
from math import ceil
from typing import Tuple
from lightgrad.grad import Tensor

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

def _fetch(url):
  import requests, os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())    
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return dat

class MNIST_Train(Dataset):
    def __init__(self, n:int =60_000, **kwargs):
        import gzip
        parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
        X = Tensor(parse(_fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))[0x10:0x10 + n*28*28].reshape((-1, 28, 28)) / 255, requires_grad=False)
        Y = Tensor(parse(_fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))[8:8+n], dtype=np.int16, requires_grad=False)
        Dataset.__init__(self, (X, Y), **kwargs)

class MNIST_Test(Dataset):
    def __init__(self, n:int =10_000, **kwargs):
        import gzip
        parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
        X = Tensor(parse(_fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"))[0x10:0x10 + n*28*28].reshape((-1, 28, 28)) / 255, requires_grad=False)
        Y = Tensor(parse(_fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))[8:8+n], dtype=np.int16, requires_grad=False)
        Dataset.__init__(self, (X, Y), **kwargs)
