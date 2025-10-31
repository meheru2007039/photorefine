"""
Watershed-based color segmentation for glare and reflection detection
"""

import numpy as np
import cv2
from scipy import ndimage


class WatershedSegmentation:
    """
    Watershed algorithm for color segmentation and mask generation
    Effective for separating regions based on color similarity
    """

    @staticmethod
    def watershed_segmentation(image, n_markers=5, compactness=0.001):
        """
        Perform watershed segmentation to create a mask of selected colors

        Args:
            image: Input image (RGB)
            n_markers: Number of marker regions
            compactness: Balance between color similarity and spatial proximity (SLIC parameter)

        Returns:
            Binary mask of brightest/most saturated regions
        """
        # Convert to LAB color space for better color segmentation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Compute markers using SLIC superpixels
        from skimage.segmentation import slic, mark_boundaries
        segments = slic(image, n_segments=n_markers * 10, compactness=compactness,
                       start_label=1, channel_axis=2)

        # Create markers from segments
        markers = np.zeros_like(segments)
        for i in range(1, segments.max() + 1):
            segment_mask = segments == i
            # Get mean brightness of segment
            brightness = np.mean(image[segment_mask])
            if brightness > 200:  # Bright segments
                markers[segment_mask] = 1

        # If no bright regions found, use threshold
        if markers.max() == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, markers = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            markers = markers.astype(np.int32)
            markers[markers > 0] = 1

        # Prepare for watershed
        markers = markers.astype(np.int32)
        markers[markers == 0] = 2  # Background
        markers[markers == 1] = 1  # Foreground (bright regions)

        # Apply watershed
        markers = cv2.watershed(image, markers)

        # Create mask from watershed results
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[markers == 1] = 255

        return mask

    @staticmethod
    def color_watershed_with_selection(image, seed_points, color_tolerance=30):
        """
        Watershed segmentation with user-selected seed points

        Args:
            image: Input image (RGB)
            seed_points: List of (x, y) coordinates for seed points
            color_tolerance: Color tolerance for region growing

        Returns:
            Binary mask of regions matching seed colors
        """
        h, w = image.shape[:2]
        markers = np.zeros((h, w), dtype=np.int32)

        # Create markers from seed points
        for i, (x, y) in enumerate(seed_points, start=1):
            cv2.circle(markers, (int(x), int(y)), 5, i, -1)

        # Background marker
        markers[markers == 0] = len(seed_points) + 1

        # Apply watershed
        markers = cv2.watershed(image, markers)

        # Create combined mask for all seed regions
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, len(seed_points) + 1):
            mask[markers == i] = 255

        return mask

    @staticmethod
    def adaptive_watershed(image, brightness_threshold=200, min_region_size=100):
        """
        Adaptive watershed that automatically detects bright regions

        Args:
            image: Input image (RGB)
            brightness_threshold: Threshold for detecting bright regions
            min_region_size: Minimum size of regions to keep

        Returns:
            Binary mask of detected regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find bright regions
        _, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Distance transform for watershed
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Find peaks (sure foreground)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Find sure background
        sure_bg = cv2.dilate(binary, kernel, iterations=3)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(image, markers)

        # Create final mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[markers > 1] = 255

        # Filter by region size
        labeled, num_features = ndimage.label(mask)
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            if np.sum(region_mask) < min_region_size:
                mask[region_mask] = 0

        return mask

    @staticmethod
    def hsv_watershed_segmentation(image, h_range=(15, 45), s_min=80, v_min=100):
        """
        Watershed segmentation based on HSV color range

        Args:
            image: Input image (RGB)
            h_range: Hue range (for glare/reflection, often in yellow-white range)
            s_min: Minimum saturation
            v_min: Minimum value (brightness)

        Returns:
            Binary mask of regions in specified HSV range
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create initial mask based on HSV thresholds
        lower = np.array([h_range[0], s_min, v_min])
        upper = np.array([h_range[1], 255, 255])
        initial_mask = cv2.inRange(hsv, lower, upper)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel)
        initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)

        # Distance transform
        dist_transform = cv2.distanceTransform(initial_mask, cv2.DIST_L2, 5)

        # Threshold distance transform
        _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Sure background
        sure_bg = cv2.dilate(initial_mask, kernel, iterations=3)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(image, markers)

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[markers > 1] = 255

        return mask

    @staticmethod
    def multi_color_watershed(image, color_ranges):
        """
        Watershed segmentation for multiple color ranges

        Args:
            image: Input image (RGB)
            color_ranges: List of HSV color ranges [(h_min, h_max, s_min, v_min), ...]

        Returns:
            Combined binary mask of all specified color regions
        """
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for h_min, h_max, s_min, v_min in color_ranges:
            mask = WatershedSegmentation.hsv_watershed_segmentation(
                image, (h_min, h_max), s_min, v_min
            )
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask
