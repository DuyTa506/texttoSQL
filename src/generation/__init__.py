"""
Generation module – SQL inference from trained models.

Training components (SFT, RL, reward, data formatting) are in the
separate `training/` folder for easier iteration and modification.

This module focuses on inference only.
"""

from .inference import SQLInference

__all__ = ["SQLInference"]
