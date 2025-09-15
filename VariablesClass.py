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


# ===================================================
# Class - User Variables: stiffnesses, lengths, etc.
# ===================================================


class VariablesClass:
    """
    Class with variables dictated by supervisor
    """
    def __init__(self, k_stiff: NDArray[np.float_], k_soft: NDArray[np.float_], thresh: NDArray[np.float_],
                 theta_ss: NDArray[np.float_], problem: str, L: float, supress_prints=True) -> None:
        self.k_stiff = k_stiff  # shim stiffness in stiff direction
        self.k_soft = k_soft  # shim stiffness in soft direction
        self.thresh = thresh
        self.theta_ss = theta_ss
        self.L = L
        self.problem = problem
        self.hinges, self.shims = np.shape(k_stiff)  # N hinges, N shims per hinge
        self.supress_prints = supress_prints
        
        # correct for un-physical threshold to move shim
        self.thresh[self.thresh < self.theta_ss] = self.theta_ss[self.thresh < self.theta_ss]
        
        # make sure soft stiffnesses are always softer than stiff
        self.k_soft[self.k_soft > self.k_stiff] = self.k_stiff[self.k_soft > self.k_stiff]
    
    def set_normalizations(self, k_stiff: NDArray[np.float_], k_soft: NDArray[np.float_], theta_ss: NDArray[np.float_]) -> None:
        self.k_bar = np.mean([np.mean(k_stiff, axis=1), np.mean(k_soft, axis=1)], axis=0)
        self.theta_bar = np.mean(theta_ss, axis=1)
