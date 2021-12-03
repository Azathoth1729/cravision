import cranet

import numpy as np
from .vision import VisionDataset
from pathlib import Path

from typing import (
    Any,
    Tuple,
    Optional,
    Callable
)

from .utils import verify_str_arg


class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        mode (str):
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    train_img = 'train-images-idx3-ubyte'
    train_lab = 'train-labels-idx1-ubyte'
    test_img = 't10k-images-idx3-ubyte'
    test_lab = 't10k-labels-idx1-ubyte'

    def __init__(self, root: Path, mode: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.mode = mode

        self.images, self.labels = self._load_data(self.raw_folder)

    @property
    def raw_folder(self) -> Path:
        return self.root / self.__class__.__name__ / 'raw'

    def _load_data(self, data_dir: Path) -> Tuple[list, list]:
        if self.mode == 'train':
            image_file = data_dir / self.train_img
            label_file = data_dir / self.train_lab
        elif self.mode == 'test':
            image_file = data_dir / self.test_img
            label_file = data_dir / self.test_lab
        else:
            raise RuntimeError('mode must be train or test')

        images = []
        labels = []

        with open(image_file, 'rb') as f:
            f.read(4)  # magic
            self.size = int.from_bytes(f.read(4), "big")
            r = int.from_bytes(f.read(4), "big")
            c = int.from_bytes(f.read(4), "big")
            for _ in range(self.size):
                mat = []
                for i in range(r):
                    mat.append([])
                    for _ in range(c):
                        mat[i].append(int.from_bytes(f.read(1), "big"))
                images.append(np.array(mat))

        with open(label_file, 'rb') as f:
            f.read(4)  # magic
            sz = int.from_bytes(f.read(4), "big")  # size
            assert self.size == sz
            for _ in range(self.size):
                label = np.array(int.from_bytes(f.read(1), "big"))
                labels.append(label)

        return images, labels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = self.images[idx]
        label = self.labels[idx]

        img = img.reshape(28 * 28)
        label = cranet.as_tensor(label)

        if self.transform is not None:
            img = self.transform(img)
        if self.transform_target is not None:
            label = self.transform_target(label)

        return img, label
