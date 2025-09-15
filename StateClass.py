import numpy as np
import copy

from numpy.typing import NDArray
from typing import TYPE_CHECKING

import funcs_physical

if TYPE_CHECKING:
    from VariablesClass import VariablesClass
    from SupervisorClass import SupervisorClass

# ===================================================
# Class - State Variables - angles theta, measured forces tau, etc.
# ===================================================


class StateClass:
    """
    Class with state variables
    """
    def __init__(self, Variabs: "VariablesClass", Supervisor: "SupervisorClass", buckle: NDArray[np.float_] = None) -> None:
        # self.buckle_in_t = np.zeros([Supervisor.iterations, Variabs.N_springs])
        # self.tau_in_t = np.zeros(Supervisor.iterations)
        self.tau = np.zeros(Variabs.hinges)
        self.buckle_in_t = []
        self.tau_in_t = []
        
        if buckle is not None:
            self.buckle = buckle
        else:
            self.buckle = np.ones((Variabs.hinges, Variabs.shims))
        if not Variabs.supress_prints:
            print('buckle pattern ', self.buckle)
        self.buckle_in_t.append(self.buckle)

    def calc_tau(self, Variabs: "VariablesClass", thetas: NDArray[np.float_], hinge: int) -> None:
        self.tau = funcs_physical.tau_hinge(thetas[hinge], self.buckle[hinge], Variabs.theta_ss[hinge], Variabs.k_stiff[hinge],
                                            Variabs.k_soft[hinge]) 
        if not Variabs.supress_prints:
            print('tau ', self.tau)
            
    def calc_Fy(self, Variabs: "VariablesClass", thetas: NDArray[np.float_]) -> None:
        self.taus = np.zeros(Variabs.hinges)
        for j, theta in enumerate(thetas):
            self.taus[j] = funcs_physical.tau_hinge(theta, self.buckle[j], Variabs.theta_ss[j],
                                                    Variabs.k_stiff[j], Variabs.k_soft[j])
        self.Fy = funcs_physical.Fy(thetas, self.taus) 
        if not Variabs.supress_prints:
            print('Fy ', self.Fy)

    def evolve_material(self, Supervisor, Variabs):
        buckle_nxt = np.zeros((Variabs.hinges, Variabs.shims))
        for i in range(Variabs.hinges):
            for j in range(Variabs.shims):
                if self.buckle[i, j] == 1 and Supervisor.input_update > Variabs.thresh[i, j]:  # buckle left
                    buckle_nxt[i, j] = -1
                elif self.buckle[i, j] == -1 and Supervisor.input_update < -Variabs.thresh[i, j]:  # buckle right
                    buckle_nxt[i, j] = 1
                else:
                    buckle_nxt[i, j] = self.buckle[i, j]
        self.buckle = copy.copy(buckle_nxt)
        self.buckle_in_t.append(self.buckle)
        if not Variabs.supress_prints:
            print('buckle pattern ', self.buckle)
