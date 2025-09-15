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
    def __init__(self, desired_buckle: NDArray[np.float_], alpha: float) -> None:
        self.desired_buckle = desired_buckle
        self.alpha = alpha
        self.input_in_t = []
        self.input_update_in_t = []
        self.loss_in_t = []
        self.loss_MSE_in_t = []        
        
        self.input_update = 0
        self.input_update_in_t.append(self.input_update)
        
    def init_dataset(self, Variabs, iterations: int, problem: str) -> None:
        self.iterations = iterations
        # self.theta_in_t = np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
        # (hinges, iterations)
        self.theta_in_t = np.random.uniform(-90, 90, (iterations, Variabs.hinges))
        if problem == 'Fy':
            # x, y coords from thetas
            self.pos_in_t = funcs_geometry.forward_points(Variabs.L, self.theta_in_t)
    
    def desired_tau(self, Variabs: "VariablesClass"):
        # Valid for a system with a single hinge
        self.desired_tau_in_t = np.zeros(self.iterations)
        for i, theta in enumerate(self.theta_in_t):
            self.desired_tau_in_t[i] = funcs_physical.tau_hinge(Variabs, theta, self.desired_buckle)

    def desired_Fy(self, Variabs: "VariablesClass"):
        self.desired_tau_in_t = np.zeros(np.shape(self.theta_in_t))
        self.desired_Fy_in_t = np.zeros(self.iterations)
        for i, thetas in enumerate(self.theta_in_t):
            for j, theta in enumerate(thetas):
                self.desired_tau_in_t[i, j] = funcs_physical.tau_hinge(theta, self.desired_buckle[j], Variabs.theta_ss[j],
                                                                       Variabs.k_stiff[j], Variabs.k_soft[j])
            self.desired_Fy_in_t[i] = funcs_physical.Fy(thetas, self.desired_tau_in_t[i])

    def set_theta(self, Variabs: "VariablesClass", iteration: int) -> None:
        self.theta = self.theta_in_t[iteration]
        self.desired_tau = self.desired_tau_in_t[iteration]
        if not Variabs.supress_prints:
            print('thetas ', self.thetas)
        
    def set_pos(self, Variabs: "VariablesClass", iteration: int) -> None:
        self.thetas = self.theta_in_t[iteration]
        self.pos = funcs_geometry.forward_points(Variabs.L, self.thetas)
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.desired_Fy = self.desired_Fy_in_t[iteration]
        
    def calc_loss(self, Variabs: "VariablesClass", State: "StateClass") -> None:
        if Variabs.problem == 'tau':
            self.loss = funcs_ML.loss_tau(self.desired_tau, State.tau)
        elif Variabs.problem == 'Fy':
            self.loss = funcs_ML.loss_Fy(self.desired_Fy, State.Fy)
        self.loss_MSE = funcs_ML.loss_MSE(self.loss)
        self.loss_in_t.append(self.loss)
        self.loss_MSE_in_t.append(self.loss_MSE)
        if not Variabs.supress_prints:
            if Variabs.problem == 'tau':
                print('desired tau ', self.desired_tau)
            elif Variabs.problem == 'Fy':
                print('desired recation force ', self.Fy)
            print('loss ', self.loss)
            print('MSE loss ', self.loss_MSE)
        
    def calc_input_update(self, State, Supervisor, Variabs) -> Union[float, NDArray[np.float_]]:
        if Variabs.problem == 'tau':
            delta_theta = funcs_ML.input_update_theta(State.tau, self.loss, self.theta, Variabs.k_bar, Variabs.theta_bar)
            input_update_nxt = copy.copy(self.input_update) + self.alpha * delta_theta
        elif Variabs.problem == 'Fy':
            delta_pos = funcs_ML.input_update_pos(State.tau, self.loss, self.thetas, Variabs.k_bar, Variabs.theta_bar)
            input_update_nxt = copy.copy(self.input_update) + self.alpha * delta_pos
        self.input_update = input_update_nxt
        self.input_update_in_t.append(self.input_update)
        if not Variabs.supress_prints:
            if Variabs.problem == 'tau':
                print('delta theta', delta_theta)
            elif Variabs.problem == 'Fy':
                print('delta pos', delta_pos)
            print('input_update ', self.input_update)
