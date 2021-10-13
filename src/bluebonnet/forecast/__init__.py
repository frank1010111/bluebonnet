# read version from installed package
from importlib.metadata import version

__version__ = version("bluebonnet")

from .forecast import Bounds, ForecasterOnePhase
