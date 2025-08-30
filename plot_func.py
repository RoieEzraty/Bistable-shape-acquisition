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

import colors

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
    n_springs = buckle_arr.shape[1]

    # Create main grid: 3 rows (loss, buckles, input) with buckle region smaller with custom ratios
    fig = plt.figure(figsize=(7, 4 + 1.2*n_springs))  
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, n_springs*0.6, 2], figure=fig)

    # --- Loss subplot (top) ---
    ax1 = fig.add_subplot(gs[0])

    # Plot Supervisor loss on left y-axis
    ax1.plot(loss_in_t, '.', lw=2, label="Loss")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    # --- Buckle subgrid (middle) ---
    gs_buckle = gridspec.GridSpecFromSubplotSpec(n_springs, 1, subplot_spec=gs[1], hspace=0.3)
    ax_buckles = []
    for i in range(n_springs):
        ax = fig.add_subplot(gs_buckle[i])
        ax.step(time, buckle_arr[:, i], where="post",
                color=colors_lst[i+1], lw=1.5, label=f"Spring {i}")
        ax.hlines(desired_buckle[i], xmin=time[0], xmax=time[-1],
                  colors=colors_lst[i+1], linestyles="--", lw=1.2,
                  label=f"Desired {i}")
        # ax.set_ylabel("State")
        ax.set_yticks([-1, 0, 1])
        ax.legend(loc="upper right", frameon=True)        
        # Hide x ticks/labels for all but the last subplot
        if i < n_springs - 1:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_buckles.append(ax)
    ax_buckles[int(n_springs/2)].set_ylabel("State")

    # --- Input subplot (bottom) ---
    ax3 = fig.add_subplot(gs[2])

    # ax3 = ax1.twinx()
    ax3.plot(input_update_in_t, '.', lw=2, label="Loss")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Inputs for update")
    ax3.tick_params(axis="y")

    plt.tight_layout()
    plt.show()
