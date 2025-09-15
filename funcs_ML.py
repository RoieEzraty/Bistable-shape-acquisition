import numpy as np


def loss_tau(desired_tau, tau):
    return desired_tau - tau


def loss_Fy(desired_Fy, Fy):
    return desired_Fy - Fy


def loss_MSE(loss):
    return np.mean(loss**2)


def input_update_theta(tau, loss, theta, k_bar, theta_bar):
    tau_pre = -1
    theta_pre = 1
    normalization = k_bar**2*theta_bar**2
    # return (tau/k_bar-theta)*(loss)/(k_bar*theta_bar)
    # return (theta/theta_bar)*(loss)/(k_bar*theta_bar)
    # return (tau_pre*tau/k_bar+theta_pre*theta)*(loss)/(k_bar*theta_bar)
    return (tau_pre*tau)*(theta_pre*theta)*(loss)/normalization


def input_update_pos(Fy, loss, pos, k_bar, pos_bar):
    F_pre = -1
    pos_pre = 1
    normalization = k_bar**2*pos_bar**2
    return (F_pre*Fy)*(pos_pre*pos)*(loss)/normalization
