"""
Basic inpainting methods including Telea, Navier-Stokes, and other approaches
"""

import cv2
import numpy as np


class BasicInpainting:
    """Collection of basic inpainting algorithms"""

    @staticmethod
    def inpaint(image, mask, method='telea', radius=5):
        """
        Apply inpainting using the specified method

        Args:
            image: Input image (RGB)
            mask: Binary mask (255 for regions to inpaint)
            method: One of 'telea', 'ns', 'bilateral', 'morphological', 'multiscale'
            radius: Inpainting radius

        Returns:
            Inpainted image
        """
        if method == 'telea':
            return BasicInpainting.telea_inpainting(image, mask, radius)
        elif method == 'ns':
            return BasicInpainting.ns_inpainting(image, mask, radius)
        elif method == 'bilateral':
            return BasicInpainting.bilateral_reconstruction(image, mask)
        elif method == 'morphological':
            return BasicInpainting.morphological_reconstruction(image, mask)
        elif method == 'multiscale':
            return BasicInpainting.multiscale_decomposition(image, mask)
        else:
            return BasicInpainting.telea_inpainting(image, mask, radius)

    @staticmethod
    def telea_inpainting(image, mask, radius=5):
        """
        Telea's Inpainting Algorithm - Fast marching method
        Good for smaller regions
        """
        result = cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        return result

    @staticmethod
    def ns_inpainting(image, mask, radius=5):
        """
        Navier-Stokes Inpainting
        Better for larger regions and preserving structure
        """
        result = cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
        return result

    @staticmethod
    def bilateral_reconstruction(image, mask):
        """
        Bilateral filter-based reconstruction
        Reduces artifacts while preserving edges
        """
        # Apply bilateral filter to reduce flare while preserving edges
        bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=100, sigmaSpace=100)

        # Create inverse mask (areas to keep from original)
        inv_mask = cv2.bitwise_not(mask)

        # Normalize masks for blending
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
        inv_mask_3ch = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2RGB).astype(float) / 255.0

        # Blend: use filtered image in flare areas, original elsewhere
        result = (bilateral.astype(float) * mask_3ch +
                  image.astype(float) * inv_mask_3ch).astype(np.uint8)

        return result

    @staticmethod
    def morphological_reconstruction(image, mask):
        """
        Morphological operations + Gaussian blending
        """
        # Close operation to fill small gaps
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Gaussian blur the affected region heavily
        blurred = cv2.GaussianBlur(image, (51, 51), 0)

        # Create smooth transition mask
        mask_smooth = cv2.GaussianBlur(closed_mask, (7, 7), 0).astype(float) / 255.0
        mask_smooth_3ch = np.stack([mask_smooth] * 3, axis=-1)

        # Blend original and blurred based on smooth mask
        result = (image.astype(float) * (1 - mask_smooth_3ch) +
                  blurred.astype(float) * mask_smooth_3ch).astype(np.uint8)

        return result

    @staticmethod
    def multiscale_decomposition(image, mask):
        """
        Multi-scale decomposition with guided filter
        Separates base and detail, removes flare from base layer
        """
        # Convert to float
        img_float = image.astype(float) / 255.0

        # Create base layer using guided filter (approximation with bilateral)
        base = cv2.bilateralFilter(image, d=15, sigmaColor=80, sigmaSpace=80).astype(float) / 255.0

        # Detail layer
        detail = img_float - base

        # Remove flare from base layer
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
        base_filtered = cv2.GaussianBlur((base * 255).astype(np.uint8), (31, 31), 0).astype(float) / 255.0

        base_corrected = base * (1 - mask_3ch) + base_filtered * mask_3ch

        # Reconstruct with attenuated detail in flare regions
        detail_attenuated = detail * (1 - mask_3ch * 0.7)
        result = base_corrected + detail_attenuated

        # Clip and convert back
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return result
