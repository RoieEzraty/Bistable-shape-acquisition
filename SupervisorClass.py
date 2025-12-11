from __future__ import annotations
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np
import copy

import funcs_geometry, funcs_physical, funcs_ML

if TYPE_CHECKING:
    from VariablesClass import VariablesClass
    from StateClass import StateClass


# ===================================================
# Class - Supervisor Variables - Loss, update values, etc.
# ===================================================


class SupervisorClass:
    """
    Class with variables dictated by supervisor
    """
    def __init__(self, alpha: float, T: int, desired_mode: str, desired_buckle: Optional[NDArray[np.float_]] = None,
                 tau0: Optional[float] = None, tau1: Optional[float] = None, beta: Optional[float] = None,
                 theta0: Optional[float] = None) -> None:
        self.T = T 
        self.desired_buckle = desired_buckle
        self.alpha = alpha
        self.input_in_t = np.zeros([T,])
        self.input_update_in_t = np.zeros([T,])
        self.loss_in_t = np.zeros([T,])
        self.loss_norm_in_t = np.zeros([T,])
        self.loss_MSE_in_t = np.zeros([T,])        
        
        self.input_update = 0
        self.input_update_in_t[0] = self.input_update

        self.desired_mode = desired_mode
        if desired_mode == 'analytic_function':
            self.desired_tau_func = lambda theta: tau0 + tau1 * np.exp(-beta*(theta-theta0))

    def init_dataset(self, Variabs, problem: str) -> None:
        # self.theta_in_t = np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
        # (hinges, iterations)
        self.theta_in_t = np.random.uniform(-90, 90, (self.T, Variabs.hinges))
        if problem == 'Fy':
            # x, y coords from thetas
            self.pos_in_t = funcs_geometry.forward_points(Variabs.L, self.theta_in_t)
    
    def desired_tau(self, Variabs: "VariablesClass"):
        # Valid for a system with a single hinge
        self.desired_tau_in_t = np.zeros(self.T)
        for i, thetas in enumerate(self.theta_in_t):
            for j, theta in enumerate(thetas):
                if self.desired_mode == 'specific_buckle':
                    self.desired_tau_in_t[i] = funcs_physical.tau_hinge(theta, self.desired_buckle, Variabs.theta_ss,
                                                                        Variabs.k_stiff, Variabs.k_soft, hinge=j)
                elif self.desired_mode == 'analytic_function':
                    self.desired_tau_in_t[i] = self.desired_tau_func(theta)

    def desired_Fy(self, Variabs: "VariablesClass"):
        self.desired_tau_in_t = np.zeros(np.shape(self.theta_in_t))
        self.desired_Fy_in_t = np.zeros(self.T)
        for i, thetas in enumerate(self.theta_in_t):
            for j, theta in enumerate(thetas):
                print('theta', theta)
                self.desired_tau_in_t[i, j] = funcs_physical.tau_hinge(theta, self.desired_buckle, Variabs.theta_ss,
                                                                       Variabs.k_stiff, Variabs.k_soft, hinge=j)
            self.desired_Fy_in_t[i] = funcs_physical.Fy(thetas, self.desired_tau_in_t[i])

    def set_theta(self, Variabs: "VariablesClass", t: int) -> None:
        self.theta = self.theta_in_t[t]
        self.desired_tau = self.desired_tau_in_t[t]
        if not Variabs.supress_prints:
            print('theta ', self.theta)
        
    def set_pos(self, Variabs: "VariablesClass", t: int) -> None:
        self.thetas = self.theta_in_t[t]
        self.pos = funcs_geometry.forward_points(Variabs.L, self.thetas)
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.desired_Fy = self.desired_Fy_in_t[t]
        
    def calc_loss(self, Variabs: "VariablesClass", State: "StateClass", t: int) -> None:
        if Variabs.problem == 'tau':
            self.loss = funcs_ML.loss_tau(self.desired_tau, State.tau)
        elif Variabs.problem == 'Fy':
            self.loss = funcs_ML.loss_Fy(self.desired_Fy, State.Fy)
        self.loss_norm = self.loss / self.theta_in_t[t]
        self.loss_MSE = funcs_ML.loss_MSE(self.loss_norm)
        self.loss_in_t[t] = self.loss
        self.loss_norm_in_t[t] = self.loss_norm
        self.loss_MSE_in_t[t] = self.loss_MSE
        if not Variabs.supress_prints:
            if Variabs.problem == 'tau':
                print('desired tau ', self.desired_tau)
            elif Variabs.problem == 'Fy':
                print('desired recation force ', self.Fy)
            print('normalized loss', self.loss_norm)
            print('MSE loss ', self.loss_MSE)
        
    def calc_input_update(self, State, Supervisor, Variabs, t: int) -> Union[float, NDArray[np.float_]]:
        if Variabs.problem == 'tau':
            delta_theta = funcs_ML.input_update_theta(State.tau, self.loss, self.theta, Variabs.k_bar, Variabs.theta_bar)
            input_update_nxt = copy.copy(self.input_update) + self.alpha * delta_theta
        elif Variabs.problem == 'Fy':
            delta_pos = funcs_ML.input_update_pos(State.tau, self.loss, self.thetas, Variabs.k_bar, Variabs.theta_bar)
            input_update_nxt = copy.copy(self.input_update) + self.alpha * delta_pos
        self.input_update = input_update_nxt
        self.input_update_in_t[t] = float(self.input_update)
        if not Variabs.supress_prints:
            if Variabs.problem == 'tau':
                print('delta theta', delta_theta)
            elif Variabs.problem == 'Fy':
                print('delta pos', delta_pos)
            print('input_update ', self.input_update)
