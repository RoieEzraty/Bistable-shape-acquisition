import numpy as np
import copy

from numpy.typing import NDArray
from typing import TYPE_CHECKING
from config import ExperimentConfig

import funcs_physical

if TYPE_CHECKING:
    from VariablesClass import VariablesClass
    from SupervisorClass import SupervisorClass
    from StructureClass import StructureClass


# ===================================================
# Class - State Variables - angles theta, measured forces tau, etc.
# ===================================================


class StateClass:
    """
    Class with state variables
    """
    def __init__(self, CFG: ExperimentConfig, Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass",
                 buckle: str = 'from config') -> None:
        # self.buckle_in_t = np.zeros([Supervisor.iterations, Variabs.N_springs])
        # self.tau_in_t = np.zeros(Supervisor.iterations)
        self.tau = np.zeros([Strctr.H,])
        self.buckle_in_t = np.zeros([Sprvsr.T, Strctr.H, Strctr.S])
        self.tau_in_t = np.zeros([Sprvsr.T,])
        
        if buckle == 'from config':
            self.buckle = CFG.State.init_buckle
        elif buckle == 'ones':
            self.buckle = np.ones((Strctr.hinges, Strctr.shims))
        if not Variabs.supress_prints:
            print('buckle pattern ', self.buckle)
        self.buckle_in_t[0, :, :] = self.buckle

    def calc_tau(self, Variabs: "VariablesClass", thetas: NDArray[np.float_], hinge: int) -> None:
        self.tau = funcs_physical.tau_hinge(thetas[hinge], self.buckle, Variabs.theta_ss, Variabs.k_stiff,
                                            Variabs.k_soft, hinge=hinge) 
        if not Variabs.supress_prints:
            print('tau ', self.tau)
            
    def calc_Fy(self, Variabs: "VariablesClass", thetas: NDArray[np.float_], hinge: int) -> None:
        self.taus = np.zeros(Variabs.hinges)
        for j, theta in enumerate(thetas):
            self.taus[j] = funcs_physical.tau_hinge(theta, self.buckle, Variabs.theta_ss,
                                                    Variabs.k_stiff, Variabs.k_soft, hinge=hinge)
        self.Fy = funcs_physical.Fy(thetas, self.taus) 
        if not Variabs.supress_prints:
            print('Fy ', self.Fy)

    def evolve_material(self, Variabs: "VariablesClass", Sprvsr: "SupervisorClass"):
        buckle_nxt = np.zeros((Variabs.hinges, Variabs.shims))
        for i in range(Variabs.hinges):
            for j in range(Variabs.shims):
                if self.buckle[i, j] == 1 and Sprvsr.input_update > Variabs.thresh[i, j]:  # buckle left
                    buckle_nxt[i, j] = -1
                elif self.buckle[i, j] == -1 and Sprvsr.input_update < -Variabs.thresh[i, j]:  # buckle right
                    buckle_nxt[i, j] = 1
                else:
                    buckle_nxt[i, j] = self.buckle[i, j]
        self.buckle = copy.copy(buckle_nxt)
        self.buckle_in_t.append(self.buckle)
        if not Variabs.supress_prints:
            print('buckle pattern ', self.buckle)
