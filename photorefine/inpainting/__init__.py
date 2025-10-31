"""
Inpainting algorithms for image restoration
"""

from .basic import BasicInpainting
from .patchmatch import PatchMatchInpainting

__all__ = ['BasicInpainting', 'PatchMatchInpainting']
