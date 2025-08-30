from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List, Dict, Any, Union, Optional
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from Network_State import Network_State
    from Big_Class import Big_Class

import colors, statistics

# ================================
# functions for plots
# ================================


