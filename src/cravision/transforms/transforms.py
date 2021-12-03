import cranet
from cranet import Tensor
from . import functional as F
from typing import (
    Any,
    Callable,
    Iterable,
)

TransformType = Callable[[Any], Any]


class Compose:
    """
    Make transforms work like composition of functions
    """

    def __init__(self, transforms: Iterable[TransformType]) -> None:
        self.transforms = transforms

    def __call__(self, img: Any) -> Any:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        fmt_str = self.__class__.__name + '('
        for t in self.transforms:
            fmt_str += '\n'
            fmt_str += f"   {t}"
        fmt_str += '\n)'
        return fmt_str


class ToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic: Union[Image, np.ndarray]) -> Tensor:
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Lambda:
    """Apply a user-defined lambda function as a transform.

    Args:
        fc (function): Lambda/function to be used for transform.
    """

    def __init__(self, fc: TransformType) -> None:
        if not callable(fc):
            raise TypeError(f"Argument fc should be callable, got {type(fc).__name__}")
        self.fc = fc

    def __call__(self, img: Any) -> Any:
        return self.fc(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
