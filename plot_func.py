from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List, Dict, Any, Union, Optional
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from Network_State import Network_State
    from Big_Class import Big_Class

import colors, funcs_geometry

# ================================
# functions for plots
# ================================


def importants(buckle_in_t: NDArray[np.int], desired_buckle: NDArray[np.int], loss_in_t: NDArray[np.float],
               input_update_in_t: NDArray[np.float]):

    # Set the custom color cycle globally without cycler
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    buckle_arr = np.asarray(buckle_in_t)
    time = np.arange(buckle_arr.shape[0])
    n_springs = buckle_arr.shape[2]

    # Create main grid: 3 rows (loss, buckles, input) with buckle region smaller with custom ratios
    fig = plt.figure(figsize=(4.6, 2.2 + 1.2*n_springs))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, n_springs*0.75, 1.2], figure=fig)

    # --- Loss subplot (top) ---
    ax1 = fig.add_subplot(gs[0])

    # Plot Supervisor loss on left y-axis
    ax1.plot(loss_in_t, '.', lw=2, label="Loss")
    ax1.set_yscale('log')
    ax1.set_ylim([1e-1, 1e3])
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")
    ax1.xaxis.set_visible(False)  # hide entire x-axis to free space

    # --- Buckle subgrid (middle) ---
    gs_buckle = gridspec.GridSpecFromSubplotSpec(n_springs, 1, subplot_spec=gs[1], hspace=0.2)
    ax_buckles = []
    for i in range(n_springs):
        ax = fig.add_subplot(gs_buckle[i])
        ax.step(time, buckle_arr[:, :, i], where="post",
                color=colors_lst[(i+1) % len(colors_lst)], lw=1.5, label=f"Spring {i}")
        if desired_buckle is not None:
            ax.hlines(desired_buckle[:, i], xmin=time[0], xmax=time[-1],
                      colors=colors_lst[(i+1) % len(colors_lst)], linestyles="--", lw=1.2)
        ax.set_yticks([-1, 0, 1])
        ax.legend(loc="upper right", frameon=True)

        # Hide entire x-axis for all but the last buckle subplot to reclaim the space
        if i < n_springs - 1:
            ax.xaxis.set_visible(False)

        ax_buckles.append(ax)

    # Put a shared y-label roughly in the middle buckle axis
    if n_springs > 0:
        ax_buckles[int(n_springs/2)].set_ylabel("State")

    # --- Input subplot (bottom) ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(input_update_in_t, '.', lw=2, label="Loss")
    ax3.set_xlabel(r'$t$')
    ax3.set_ylabel(r'$\theta^!$')
    ax3.tick_params(axis="y")

    plt.tight_layout()
    plt.show()


def plot_response(theta: NDArray[np.float_], tau_init: float, tau_fin: float, tau_des: float, theta_range=[-180, 180],
                  just_init=False) -> None:
    """
    tau a.f.o theta
    """

    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    # Plot
    # initial tau a.f.o theta
    ax.plot(theta, tau_init, lw=2, color=colors_lst[0])
    legend = [r'$\tau\left(0\right)$', r'$\hat{\tau}$']
    if not just_init:  # plot also final state
        legend = [r'$\tau\left(0\right)$', r'$\tau\left(end\right)$', r'$\hat{\tau}$']
        # final tau a.f.o theta
        ax.plot(theta, tau_fin, lw=2, color=colors_lst[2])
    # desired tau a.f.o theta
    ax.plot(theta, tau_des, '--', lw=2, color=colors_lst[1])
    
    # beautify
    ax.legend(legend)
    if theta_range:
        ax.set_xlim(theta_range)
    ax.set_ylim([-5000, 5000])
    ax.set_ylabel(r'$\tau$')
    ax.set_xlabel(r'$\theta$')
    plt.tight_layout()
    plt.show()


def plot_arm(pos_vec, theta_vec, L) -> None:
    """
    geometrical snapshot of arm with angles and x-y coords of tip
    """
    p0 = [0, 0]
    p1 = pos_vec[0, :]
    p2 = pos_vec[1, :]
    theta1 = np.deg2rad(theta_vec[0])
    theta2 = np.deg2rad(theta_vec[1])
    y_wall = p2[1]
    
    # Set the custom color cycle globally without cycler
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)
    
    # p0, p1, p2 = funcs_geometry.forward_points(L, theta1, theta2)
    # N, tau1, tau2 = normal_force(L q1, q2, k1, k2, th1_ss, th2_ss)

    # Figure
    plt.figure(figsize=(4, 4))
    # Wall (horizontal at y=y_wall) and fixed x (vertical at x=x_star)
    xs = np.linspace(-2*L-0.5, 2*L+0.5, 2)
    plt.plot(xs, y_wall*np.ones_like(xs), linewidth=2, linestyle='--', label="wall y=y_w")

    # Links
    plt.plot([p0[0], p0[0]], [-L/3, p0[1]], '-k', linewidth=4)  # straight line up to first node 
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=4)
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=4)

    # Joints and tip
    plt.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], s=60, zorder=3)
    # Annotate tip and normal force arrow
    plt.annotate("Tip", xy=(p2[0], p2[1]), xytext=(p2[0]+0.05, p2[1]+0.05))

    # Angle markers (small arcs)
    # base angle q1
    a = np.linspace(-np.pi/2, theta1+np.pi/2, 60)
    r = 0.15*L
    plt.plot(r*np.cos(a), r*np.sin(a))
    # elbow relative angle q2 centered at p1
    a2 = np.linspace(-np.pi/2+theta1, theta1+theta2+np.pi/2, 60)
    r2 = 0.15*L 
    plt.plot(p1[0] + r2*np.cos(a2), p1[1] + r2*np.sin(a2))
    plt.text(p1[0]+0.05, p1[1]+0.05, r"$\theta_2$")
    plt.text(0.05, 0.05, r"$\theta_1$")

    # Aesthetics
    plt.axis('equal')
    reach = 2*L + 0.4
    plt.xlim(-reach, reach)
    plt.ylim(-reach, reach)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Tip (x, y)=({p2[0]:.2f}, {p2[1]:.2f})")
    plt.grid(True)
    # plt.legend(loc='lower left')
    plt.show()
