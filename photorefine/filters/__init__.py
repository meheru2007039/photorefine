"""
Advanced filtering algorithms for glare and reflection detection
"""

from .color_segmentation import WatershedSegmentation
from .fourier import DFTFilter

__all__ = ['WatershedSegmentation', 'DFTFilter']
