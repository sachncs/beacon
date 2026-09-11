"""beacon — training-free Sliding Window Attention with sinks.

Reproduction of "Sliding-window beats linear attention"
(Jolicoeur-Martineau et al., 2026).

Public API::

    from beacon import Config, patch, mask, load
"""

from .load import load
from .patch import Config, mask, patch, unpatch

__version__ = "0.2.0"
__all__ = ["Config", "mask", "patch", "unpatch", "load", "__version__"]