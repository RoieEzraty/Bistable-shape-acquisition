from __future__ import annotations
from dataclasses import dataclass

import numpy as np


# -----------------------------
# Structure and initial params
# -----------------------------
@dataclass(frozen=True)
class StructureConfig:
    H: int = 1  # Hinges
    S: int = 20  # Shims per hinge
    L: float = 1.  # length of edge (2 per hinge)
    Nin: int = 3  # tip position in (x, y) and its angle
    Nout: int = 3  # Fx, Fy, torque, all on tip


# -----------------------------
# Material / variables
# -----------------------------
@dataclass(frozen=True)
class VariablesConfig:
    # k_stiff = np.array([2, 1])
    # k_soft = np.array([1, 0.5])
    # thresh = np.array([40, 20])
    # theta_ss = np.array([30, 10])
    # desired_buckle = np.array([1, -1])

    # k_stiff = np.array([[10, 8]])
    # k_soft = np.array([[1, 0.01]])
    # thresh = np.array([[40, 20]])
    # theta_ss = np.array([[30, 10]])
    # desired_buckle = np.array([[1, -1]])

    # k_stiff = np.array([[4.2, 2.0, 1.0]])
    # k_soft = np.array([[0.2, 0.5, 0.1]])
    # thresh = np.array([[30, 25, 15]])
    # theta_ss = np.array([[5, 15, 10]])
    # desired_buckle = np.array([[1, 1, -1]])

    # k_stiff: tuple = (6.2, 5.5, 5.0, 3.9, 3.8, 3.7, 3.6, 3.5, 2.0, 1.0)
    # k_soft: tuple = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.1)
    # thresh: tuple = (125, 120, 115, 110, 105, 100, 95, 90, 85, 80)
    # theta_ss: tuple = (60, 50, 39, 24, 33, 28, 13, 11, 10, 9)

    k_stiff: tuple = tuple(np.linspace(40, 4, 20))
    # k_stiff: tuple = tuple(np.linspace(20, 4, 10))

    # k_soft: tuple = tuple(np.linspace(3, 0.03, 20))
    k_soft: tuple = tuple(np.flip(np.linspace(3, 0.3, 10)))

    thresh: tuple = tuple(np.linspace(95, 165, 20))
    # thresh: tuple = tuple(np.flip(np.linspace(100, 160, 10)))

    theta_ss: tuple = tuple(np.linspace(5, 65, 20))
    # theta_ss: tuple = tuple(np.flip(np.linspace(5, 60, 10)))

    supress_prints: bool = False


# -----------------------------
# Training / supervisor
# -----------------------------
@dataclass(frozen=True)
class TrainingConfig:
    T: int = 2000  # total training set time (not time to reach equilibrium during every step)
    alpha: float = 0.02  # learning rate

    problem: str = 'tau'
    # problem: str = 'Fy'

    desired_mode: str = 'analytic_function'
    # desired_mode: str = 'specific_buckle'

    tau0: float = +2000
    tau1: float = 0.7
    beta: float = 0.065
    theta0: float = 65

    desired_buckle: tuple = (1, 1, -1, 1, 1, -1, 1, 1, -1, 1)

    rand_key_dataset: int = 8  # for random sampling of dataset, if dataset_sampling is True


# -----------------------------
# State
# -----------------------------
@dataclass(frozen=True)
class StateConfig:

    # init_buckle: tuple = (1, -1, -1, 1, -1, -1, 1, 1, 1, 1)
    init_buckle: tuple = (1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1)
    # init_buckle: tuple = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)  # ones


# -----------------------------
# A single top-level config
# -----------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    Strctr: StructureConfig = StructureConfig()
    Variabs: VariablesConfig = VariablesConfig()
    Train: TrainingConfig = TrainingConfig()
    State: StateConfig = StateConfig()


# Default instance you can import directly
CFG = ExperimentConfig()
