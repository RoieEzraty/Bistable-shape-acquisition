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
    from StructureClass import StructureClass
    from config import ExperimentConfig


# ===================================================
# Class - User Variables: stiffnesses, lengths, etc.
# ===================================================


class VariablesClass:
    """
    Class with variables dictated by supervisor.
    The shims used have a bilinear stress-strain relation with a stiff and a soft direction around a non-zero steady state angle
    at which torque is zero. Each shim buckles at a certain threshold angle, whereupon the soft and stiff directions, as well
    as the steady state angle, flip. So each shim in each hinge has 4 variables - soft stiffness k_soft, stiff stiffness k_stiff, 
    steady state angle theta_ss and threshold angle thresh.

    Attributes
    ----------
    k_stiff        - (H, S) NDArray of the stiff stiffness for each shim
    k_soft         - (H, S) NDArray of the soft stiffness for each shim
    thresh         - (H, S) NDArray of thershold angle (degs) at which each shim buckles
    theta_ss       - (H, S) NDArray of steady state angle (degs) of shims, at which no torque is measured. It is usually not zero.

    supress_prints - bool, whether to suppress printing all the variables as they are calculated 
                     True = don't print
    """
    k_stiff: NDArray
    k_soft: NDArray
    thresh: NDArray
    theta_ss: NDArray
    supress_prints: bool

    def __init__(self, CFG: ExperimentConfig, Strctr: StructureClass) -> None:
        """
        Parameters
        ----------
        CFG    - ExperimentConfig.
        Strctr - StructureClass.
        """
        self.k_stiff = Strctr._custom_reshape(CFG.Variabs.k_stiff)  # shim stiffness in stiff direction
        self.k_soft = Strctr._custom_reshape(CFG.Variabs.k_soft)  # shim stiffness in soft direction
        self.thresh = Strctr._custom_reshape(CFG.Variabs.thresh)
        self.theta_ss = Strctr._custom_reshape(CFG.Variabs.theta_ss)
        self.supress_prints = CFG.Variabs.supress_prints
        
        # correct for un-physical threshold to move shim
        if (self.thresh < self.theta_ss).any():
            print('corrected for threshold lower than steady state theta')
            print('there was one in idx', np.where(self.thresh < self.theta_ss))
            self.thresh[self.thresh < self.theta_ss] = self.theta_ss[self.thresh < self.theta_ss]
        
        # make sure soft stiffnesses are always softer than stiff
        if (self.k_soft > self.k_stiff).any():
            print('corrected for soft k stiffer than stiff k')
            print('there was one in idx', np.where(self.k_soft > self.k_stiff))
            self.k_soft[self.k_soft > self.k_stiff] = self.k_stiff[self.k_soft > self.k_stiff]
    
    def set_normalizations(self) -> None:
        """
        set variables used to normalize torques in loss, etc
        """
        self.k_bar = np.mean([np.mean(self.k_stiff, axis=1), np.mean(self.k_soft, axis=1)], axis=0)
        self.theta_bar = np.mean(self.theta_ss, axis=1)
