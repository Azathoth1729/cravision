import os
import sys
import random
import unittest
import numpy as np
from pathlib import Path

import torch
import torchvision

import cranet
from src import cravision

from .utils import show_example, teq


class TestVision(unittest.TestCase):
    def test_svhn(self):
        print()
        HOME = Path.home()
        WORKSHOP = HOME / "workshop" / "python"
        DATA_DIR = WORKSHOP / "dl" / "dataset" / "svhn"

        cradataset = cravision.datasets.SVHN(root=DATA_DIR, transform=cravision.transforms.ToTensor())
        torchdataset = torchvision.datasets.SVHN(root=DATA_DIR, transform=torchvision.transforms.ToTensor())

        test_nums = 1000
        i = 0
        for tdata, cdata in zip(torchdataset, cradataset):
            if i >= test_nums:
                break

            i += 1
            torimg, _ = tdata
            cimg, clabel = cdata
            self.assertTrue(teq(torimg, cimg, 1e-7), f"torch:{torimg.detach().numpy()}\n\ncranet:{cimg.detach().numpy()}")

    def test_svhn_show(self):
        print()
        HOME = Path.home()
        DATA_DIR = HOME / "Downloads" / "dataset"
        cradataset = cravision.datasets.SVHN(root=DATA_DIR, transform=cravision.transforms.ToTensor())
        cimg, clabel = cradataset[0]
        show_example(cimg, clabel)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
