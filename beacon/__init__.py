"""beacon — training-free Sliding Window Attention with sinks.

Reproduction of "Sliding-window beats linear attention"
(Jolicoeur-Martineau et al., 2026).
"""

from .patch import patch_swa, swa_mask, SWAPatchedModel

__version__ = "0.1.0"
__all__ = ["patch_swa", "swa_mask", "SWAPatchedModel", "__version__"]