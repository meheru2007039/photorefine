"""
Advanced filtering algorithms for glare and reflection detection
"""

from .color_segmentation import WatershedSegmentation
from .fourier import DFTFilter
from .contrast import ContrastEnhancement, HistogramAnalysis

__all__ = ['WatershedSegmentation', 'DFTFilter', 'ContrastEnhancement', 'HistogramAnalysis']
