"""Ease plotting production and fluid flow information."""
from __future__ import annotations

import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np


class SquareRootScale(mscale.ScaleBase):
    """ScaleBase class for generating square root scale."""

    name = "squareroot"

    def __init__(self, axis, **kwargs):
        """Initialize for axis."""
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        """Set major and minor locators and formatters."""
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Do not allow negative values."""
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        """Transform from linear to square root position."""

        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            """Actual transform."""
            return np.array(a) ** 0.5

        def inverted(self):
            """Inverse transform."""
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        """Inverted square-root transform."""

        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            """Square everything."""
            return np.array(a) ** 2

        def inverted(self):
            """Square root it. (Inverse of inverse)."""
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        """Get square root transform."""
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)
