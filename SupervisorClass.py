from __future__ import annotations
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np
import copy

import funcs_geometry, funcs_physical, funcs_ML

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from VariablesClass import VariablesClass
    from StateClass import StateClass
    from config import ExperimentConfig


# ===================================================
# Class - Supervisor Variables - Loss, update values, etc.
# ===================================================


class SupervisorClass:
    """
    Class with variables dictated by supervisor

    Attributes
    ----------

    problem        - str, type of measurement used in problem
                 'tau' = torque is optimized
                 'Fy'  = force in y direction is optimized
    """
    problem: str

    def __init__(self, CFG: ExperimentConfig, Strctr: StructureClass) -> None:
        """
        Parameters
        ----------
        CFG    - ExperimentConfig.
        Strctr - StructureClass.
        
        """
        self.T = CFG.Train.T 
        self.rand_key_dataset = CFG.Train.rand_key_dataset
        self.problem = CFG.Train.problem
        self.desired_mode = CFG.Train.desired_mode
        if self.desired_mode == 'analytic_function':
            self.desired_buckle = None
            self.desired_tau_func = lambda theta: CFG.Train.tau0 + \
                                                  CFG.Train.tau1 * np.exp(-CFG.Train.beta * (theta - CFG.Train.theta0))
        elif self.desired_mode == 'specific_buckle':
            self.desired_buckle = Strctr._custom_reshape(CFG.Train.desired_buckle)
        self.alpha = CFG.Train.alpha
        self.input_update_in_t = np.zeros([self.T,])
        self.loss_in_t = np.zeros([self.T,])
        self.loss_norm_in_t = np.zeros([self.T,])
        self.loss_MSE_in_t = np.zeros([self.T,])        
        
        self.input_update = 0
        self.input_update_in_t[0] = self.input_update

        self.eps = CFG.Train.eps
        self.window_for_kill = CFG.Train.window_for_kill

        self.skip_to_thresh = CFG.Train.skip_to_thresh
        if self.skip_to_thresh:
            self.skip = min(CFG.Variabs.thresh)

    def init_dataset(self, Strctr: "StructureClass", Variabs: "VariablesClass",) -> None:
        # self.theta_in_t = np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
        rng = np.random.default_rng(self.rand_key_dataset)
        self.theta_in_t = rng.uniform(low=-90, high=90, size=(self.T, Strctr.H))  # (T, hinges)
        # self.theta_in_t = np.random.uniform(-90, 90, (self.T, Strctr.H))  # (T, hinges) old
        if self.problem == 'Fy':
            # x, y coords from thetas
            self.pos_in_t = funcs_geometry.forward_points(Strctr.L, self.theta_in_t)
    
    def desired_tau(self, Variabs: "VariablesClass") -> None:
        # Valid for a system with a single hinge
        self.desired_tau_in_t = np.zeros(self.T)
        for i, thetas in enumerate(self.theta_in_t):
            for j, theta in enumerate(thetas):
                if self.desired_mode == 'specific_buckle':
                    self.desired_tau_in_t[i] = funcs_physical.tau_hinge(theta, self.desired_buckle, Variabs.theta_ss,
                                                                        Variabs.k_stiff, Variabs.k_soft, hinge=j)
                elif self.desired_mode == 'analytic_function':
                    self.desired_tau_in_t[i] = self.desired_tau_func(theta)

    def set_theta(self, Variabs: "VariablesClass", t: int) -> None:
        self.theta = self.theta_in_t[t]
        self.desired_tau = self.desired_tau_in_t[t]
        if not Variabs.supress_prints:
            print('theta ', self.theta)
        
    def set_pos(self, Strctr: "StructureClass", t: int) -> None:
        self.thetas = self.theta_in_t[t]
        self.pos = funcs_geometry.forward_points(Strctr.L, self.thetas)
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.desired_Fy = self.desired_Fy_in_t[t]
        
    def calc_loss(self, Variabs: "VariablesClass", State: "StateClass", t: int) -> None:
        if self.problem == 'tau':
            self.loss = funcs_ML.loss_tau(self.desired_tau, State.tau)
        elif self.problem == 'Fy':
            self.loss = funcs_ML.loss_Fy(self.desired_Fy, State.Fy)
        # tau_norm_of_theta = Variabs.k_bar * self.theta_in_t[t]**2
        # self.loss_norm = self.loss / tau_norm_of_theta
        self.loss_norm = self.loss / Variabs.tau_bar
        self.loss_MSE = funcs_ML.loss_MSE(self.loss_norm)
        self.loss_in_t[t] = self.loss
        self.loss_norm_in_t[t] = self.loss_norm
        self.loss_MSE_in_t[t] = self.loss_MSE
        if not Variabs.supress_prints:
            if self.problem == 'tau':
                print('desired tau ', self.desired_tau)
            elif self.problem == 'Fy':
                print('desired recation force ', self.Fy)
            print('normalized loss', self.loss_norm)
            print('MSE loss ', self.loss_MSE)
        
    def calc_input_update(self, State: "StateClass", Supervisor: "SupervisorClass", Variabs: "VariablesClass",
                          t: int) -> None:
        if self.problem == 'tau':
            delta_theta = funcs_ML.input_update_theta(State.tau, self.loss_norm, self.theta, Variabs.k_bar, Variabs.theta_bar)
            input_update_nxt = copy.copy(self.input_update) + self.alpha * delta_theta
            input_update_nxt = funcs_physical.clip_theta(input_update_nxt)
        elif self.problem == 'Fy':
            delta_pos = funcs_ML.input_update_pos(State.tau, self.loss, self.thetas, Variabs.k_bar, Variabs.theta_bar)
            input_update_nxt = copy.copy(self.input_update) + self.alpha * delta_pos

        if self.skip_to_thresh:  # skip the middle angles that do nothing
            if np.sign(self.alpha * delta_theta) < 0 and self.input_update > 0 and self.input_update < self.skip:
                input_update_nxt = -self.skip
            elif np.sign(self.alpha * delta_theta) > 0 and self.input_update < 0 and self.input_update > -self.skip:
                input_update_nxt = self.skip

        self.input_update = input_update_nxt
        self.input_update_in_t[t] = float(self.input_update)
        if not Variabs.supress_prints:
            if self.problem == 'tau':
                print('delta theta', delta_theta)
            elif self.problem == 'Fy':
                print('delta pos', delta_pos)
            print('input_update ', self.input_update)

# ===================================================
# Not in use
# ===================================================

# def desired_Fy(self, Variabs: "VariablesClass") -> None:
#     self.desired_tau_in_t = np.zeros(np.shape(self.theta_in_t))
#     self.desired_Fy_in_t = np.zeros(self.T)
#     for i, thetas in enumerate(self.theta_in_t):
#         for j, theta in enumerate(thetas):
#             print('theta', theta)
#             self.desired_tau_in_t[i, j] = funcs_physical.tau_hinge(theta, self.desired_buckle, Variabs.theta_ss,
#                                                                    Variabs.k_stiff, Variabs.k_soft, hinge=j)
#         self.desired_Fy_in_t[i] = funcs_physical.Fy(thetas, self.desired_tau_in_t[i])