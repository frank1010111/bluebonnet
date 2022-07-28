"""Use the Hall-Yarbrough Z-factor approach.

Note: `z_factor_DAK` is the recommended approach, not this one.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def zfact(p: float | NDArray[np.float64], t: float) -> float | NDArray[np.float64]:
    """Get Z-factor for gas from Hall-Yarbrough's iterative approach.

    Parameters
    ----------
    p : float | NDArray[np.float64]
        pressure (psi)
    t : float
        temperature (Rankine)

    Returns
    -------
    float | NDArray[np.float64]
        z-factor

    References
    ----------
    <https://wiki.whitson.com/eos/eos_models/zfactor/index.html#hall-yarbrough-estimation-of-gas-z-factor>
    """
    t = 1 / t
    y = 0.001
    fdum = 1
    while np.abs(fdum) > 0.001:
        fdum = (
            -0.06125 * p * t * np.exp(-1.2 * (1 - t) ** 2)
            + (y + y**2 + y**3 - y**4) / (1 - y) ** 3
            - (14.76 * t - 9.76 * t**2 + 4.58 * t**3) * y**2
            + (90.7 * t - 242.2 * t**2 + 42.4 * t**3) * y ** (2.18 + 2.82 * t)
        )
        dfdy = (
            (1 + 4 * y + 4 * y**2 - 4 * y**3 + y**4) / (1 - y) ** 4
            - (29.52 * t - 19.52 * t**2 + 9.16 * t**3) * y
            + (2.18 + 2.82 * t)
            * (90.7 * t - 242.2 * t**2 + 42.4 * t**3)
            * y ** (1.18 + 2.82 * t)
        )
        y = y - fdum / dfdy
    zfact = 0.06125 * p * t * np.exp(-1.2 * (1 - t) ** 2) / y
    return zfact
