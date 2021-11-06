import numpy as np
from PIL import Image

import cranet
from cranet import Tensor

from . import functional_pil as F_pil
from . import functional_tensor as F_t

from typing import (
    Any,
)


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor."""
    if not (F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError(f"input pic should be PIL image or numpy.ndarray, Got {type(pic)}")
    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError(f"input pic should be 2 or 3 dimensional. Got {pic.ndim} dimensions")

    # handle np.ndarray
    if isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            pic = pic[:, :, None]
            img = cranet.tensor(pic.transpose(2, 0, 1))
            return img

    # handle PIL Image
    mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
    img = cranet.tensor(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8))
    )
    if pic.mode == '1':
        img = 255 * img

    img = img.reshape(pic.size[1], pic.size[0], len(pic.getbands()))
    # (H x W x C) -> (C x H x W)
    img = img.permute((2, 0, 1))
    return img / 255
