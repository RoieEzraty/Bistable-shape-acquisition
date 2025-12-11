import numpy as np
from numpy.typing import NDArray

from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from VariablesClass import VariablesClass


def Fy(thetas: NDArray[np.float_], taus: NDArray[np.float_]) -> float:
    thetas_rad = np.deg2rad(thetas)
    # denom = np.sin(thetas[1])
    return -taus[0]*np.sin(thetas_rad[0])
    # return (np.cos(thetas_rad[0])*taus[1] + np.cos(thetas_rad[0]+thetas_rad[1])*(taus[1] - taus[0])) / denom
    # return (np.sin(np.deg2rad(thetas[0]))*taus[1] + 
    #         np.sin(np.deg2rad(thetas[0])+np.deg2rad(thetas[1]))*(taus[1]-taus[0]))/np.sin(np.deg2rad(thetas[1]))
    # return (np.cos(np.deg2rad(thetas[0]))*taus[1] + 
    #         np.cos(np.deg2rad(thetas[0])+np.deg2rad(thetas[1]))*(taus[1]-taus[0]))/np.sin(np.deg2rad(thetas[1]))


def tau_hinge(theta, buckle, theta_ss, k_stiff, k_soft):
    return np.sum(tau_k(theta, buckle, theta_ss, k_stiff, k_soft))
    

def tau_k(theta, buckle, theta_ss, k_stiff, k_soft, h=0):
    tau_k = np.zeros(np.size(theta_ss))
    for i in range(np.size(theta_ss)):
        if buckle[h, i] == 1 and theta > -theta_ss[h, i] or buckle[h, i] == -1 and theta < theta_ss[h, i]:
            k = k_stiff[h, i]
        else:
            k = k_soft[h, i]
        tau_k[i] = -k * (theta - (-buckle[h, i]) * theta_ss[h, i])
    return tau_k


def measure_full_response(buckle, theta_ss, k_stiff, k_soft, length=100):
    theta_vec = np.linspace(-180, 180, length)
    tau_vec = np.zeros([length])
    for i, theta in enumerate(theta_vec):
        tau_vec[i] = tau_hinge(theta, buckle, theta_ss, k_stiff, k_soft)
    return theta_vec, tau_vec
