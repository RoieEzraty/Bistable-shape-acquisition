from __future__ import annotations
import numpy as np
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import Callable, Union, Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle
from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from config import ExperimentConfig


# ===================================================
# Class - Structure: shims, hinges, lengths
# ===================================================


class StructureClass:
    """
    Class with hinge structur variables

    Attributes
    ----------
    H            - Total number of hinges
    S            - Number of shims per hinge, the total on a hinge is the sum of torques on every shim
    L            - Length of every edge (2 per hinge)
    array_shapes - tuple, deisred shape of relevant arrays. It's (Strctr.H, Strctr.S). Used inside _custom_reshape()
    total_shims  - int, total number of shims in system, which is H*S. Used inside _custom_reshape()
    """
    H: int
    S: int
    L: float
    array_shapes: tuple
    total_shims: int

    def __init__(self, CFG: ExperimentConfig) -> None:
        """
        Parameters
        ----------
        CFG : ExperimentConfig.
        """
        self.H, self.S, self.L = CFG.Strctr.H, CFG.Strctr.S, CFG.Strctr.L

        self.array_shapes = (self.H, self.S)
        self.total_shims = self.H * self.S

    def _custom_reshape(self, tup: tuple) -> NDArray:
        """
        custom function to 
        1) reshape from tuple to array
        2) account for length - tuple is longer than array, array has the size (H, S)
        """
        return np.reshape(np.asarray(tup[:self.total_shims]), self.array_shapes)
