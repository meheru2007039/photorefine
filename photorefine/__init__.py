"""
PhotoRefine: Advanced Image Processing for Glare and Reflection Removal
"""

__version__ = "1.0.0"
__author__ = "PhotoRefine Team"

from .core.image_processor import ImageProcessor
from .core.blob_processor import BlobRemovalProcessor

__all__ = ['ImageProcessor', 'BlobRemovalProcessor']
