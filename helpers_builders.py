from __future__ import annotations

import numpy as np

from numpy.typing import NDArray


def mov_ave(data: NDArray[np.float_], window_size: int) -> NDArray[np.float_]:
    """
    Moving average filter to 1D data over window size using convolution.

    Parameters
    ----------
    data : 1D np.ndarray input data to be smoothed.
    window_size : int, number of elements to average over.

    Returns
    -------
    (len(data) - window_size + 1) np.ndarray of smoothed data array after applying moving average.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
