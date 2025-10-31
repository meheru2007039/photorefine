"""
PatchMatch inpainting algorithm (Barnes et al., 2009) with guidance support
Efficient approximate nearest neighbor field computation for image inpainting
"""

import numpy as np
import cv2
from numba import jit


class PatchMatchInpainting:
    """
    PatchMatch-based inpainting with guidance
    Based on "PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing"
    by Barnes et al., 2009
    """

    @staticmethod
    def inpaint(image, mask, guide=None, patch_size=7, iterations=5, alpha=0.5):
        """
        Inpaint image using PatchMatch algorithm

        Args:
            image: Input image (RGB)
            mask: Binary mask (255 for regions to inpaint, 0 for known regions)
            guide: Optional guidance image (e.g., edge map, structure) to constrain matching
            patch_size: Size of patches (must be odd)
            iterations: Number of PatchMatch iterations
            alpha: Guidance weight (0 = no guidance, 1 = full guidance)

        Returns:
            Inpainted image
        """
        if patch_size % 2 == 0:
            patch_size += 1

        h, w, c = image.shape
        result = image.copy()
        mask_binary = (mask > 0)

        # Create guidance map if provided
        if guide is None:
            # Use gradient magnitude as default guide
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            guide = np.sqrt(gx**2 + gy**2).astype(np.float32)
            guide = (guide / guide.max() * 255).astype(np.uint8) if guide.max() > 0 else guide.astype(np.uint8)

        # Dilate mask to get inpainting boundary
        kernel = np.ones((patch_size, patch_size), np.uint8)
        mask_dilated = cv2.dilate(mask_binary.astype(np.uint8), kernel, iterations=1)
        boundary = mask_dilated & (~mask_binary)

        # Multi-resolution inpainting
        result = PatchMatchInpainting._multiscale_inpaint(
            result, mask_binary, guide, patch_size, iterations, alpha
        )

        return result

    @staticmethod
    def _multiscale_inpaint(image, mask, guide, patch_size, iterations, alpha):
        """Multi-scale PatchMatch inpainting for better results"""
        # Build pyramid
        levels = 3
        img_pyramid = [image]
        mask_pyramid = [mask]
        guide_pyramid = [guide]

        for i in range(levels - 1):
            img_down = cv2.pyrDown(img_pyramid[-1])
            mask_down = cv2.pyrDown(mask_pyramid[-1].astype(np.float32)) > 0.5
            guide_down = cv2.pyrDown(guide_pyramid[-1])
            img_pyramid.append(img_down)
            mask_pyramid.append(mask_down)
            guide_pyramid.append(guide_down)

        # Inpaint from coarse to fine
        result = None
        for level in range(levels - 1, -1, -1):
            if result is not None:
                result = cv2.pyrUp(result, dstsize=(img_pyramid[level].shape[1], img_pyramid[level].shape[0]))
            else:
                result = img_pyramid[level].copy()

            result = PatchMatchInpainting._patchmatch_inpaint(
                result,
                mask_pyramid[level],
                guide_pyramid[level],
                patch_size,
                iterations,
                alpha
            )

        return result

    @staticmethod
    def _patchmatch_inpaint(image, mask, guide, patch_size, iterations, alpha):
        """Single-scale PatchMatch inpainting"""
        h, w, c = image.shape
        result = image.copy().astype(np.float32)
        half_patch = patch_size // 2

        # Find pixels to inpaint (target pixels)
        target_coords = np.argwhere(mask)
        if len(target_coords) == 0:
            return result.astype(np.uint8)

        # Find source pixels (known regions, away from mask)
        kernel = np.ones((patch_size * 2, patch_size * 2), np.uint8)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        source_mask = (~mask) & (~mask_dilated.astype(bool))
        source_coords = np.argwhere(source_mask)

        if len(source_coords) == 0:
            # Fallback if no valid source regions
            return cv2.inpaint(image, (mask * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)

        # Initialize nearest neighbor field randomly
        nnf = PatchMatchInpainting._initialize_nnf(target_coords, source_coords, h, w)

        # PatchMatch iterations
        for iter_num in range(iterations):
            # Propagation and random search
            nnf = PatchMatchInpainting._patchmatch_iteration(
                result, guide, mask, nnf, target_coords, source_coords,
                patch_size, alpha, iter_num
            )

        # Reconstruct image from NNF
        result = PatchMatchInpainting._reconstruct_from_nnf(
            result, mask, nnf, target_coords, patch_size
        )

        return result.astype(np.uint8)

    @staticmethod
    def _initialize_nnf(target_coords, source_coords, h, w):
        """Initialize nearest neighbor field randomly"""
        n_targets = len(target_coords)
        n_sources = len(source_coords)

        # Random initialization
        nnf = np.zeros((n_targets, 2), dtype=np.int32)
        for i in range(n_targets):
            rand_idx = np.random.randint(0, n_sources)
            nnf[i] = source_coords[rand_idx]

        return nnf

    @staticmethod
    def _patchmatch_iteration(image, guide, mask, nnf, target_coords, source_coords,
                               patch_size, alpha, iteration):
        """Single PatchMatch iteration with propagation and random search"""
        h, w, c = image.shape
        half_patch = patch_size // 2
        n_targets = len(target_coords)

        # Alternate scan direction
        if iteration % 2 == 0:
            indices = range(n_targets)
            offsets = [(-1, 0), (0, -1)]  # Propagate from top and left
        else:
            indices = range(n_targets - 1, -1, -1)
            offsets = [(1, 0), (0, 1)]  # Propagate from bottom and right

        new_nnf = nnf.copy()

        for idx in indices:
            ty, tx = target_coords[idx]
            best_match = nnf[idx]
            best_dist = PatchMatchInpainting._patch_distance(
                image, guide, ty, tx, best_match[0], best_match[1],
                patch_size, mask, alpha
            )

            # Propagation
            for dy, dx in offsets:
                neighbor_idx = idx + dy * w + dx
                if 0 <= neighbor_idx < n_targets:
                    ny, nx = target_coords[neighbor_idx]
                    if np.abs(ny - ty) + np.abs(nx - tx) <= 1:  # Check if neighbor
                        candidate = nnf[neighbor_idx] + np.array([dy, dx])
                        cy, cx = candidate
                        if 0 <= cy < h and 0 <= cx < w and not mask[cy, cx]:
                            dist = PatchMatchInpainting._patch_distance(
                                image, guide, ty, tx, cy, cx, patch_size, mask, alpha
                            )
                            if dist < best_dist:
                                best_match = candidate
                                best_dist = dist

            # Random search
            search_radius = max(h, w)
            while search_radius > 1:
                sy = best_match[0] + np.random.randint(-search_radius, search_radius + 1)
                sx = best_match[1] + np.random.randint(-search_radius, search_radius + 1)

                if 0 <= sy < h and 0 <= sx < w and not mask[sy, sx]:
                    dist = PatchMatchInpainting._patch_distance(
                        image, guide, ty, tx, sy, sx, patch_size, mask, alpha
                    )
                    if dist < best_dist:
                        best_match = np.array([sy, sx])
                        best_dist = dist

                search_radius //= 2

            new_nnf[idx] = best_match

        return new_nnf

    @staticmethod
    def _patch_distance(image, guide, ty, tx, sy, sx, patch_size, mask, alpha):
        """
        Calculate distance between patches with guidance

        Args:
            ty, tx: Target patch center
            sy, sx: Source patch center
            alpha: Weight for guidance term
        """
        h, w, c = image.shape
        half_patch = patch_size // 2

        # Extract patches
        ty_start = max(0, ty - half_patch)
        ty_end = min(h, ty + half_patch + 1)
        tx_start = max(0, tx - half_patch)
        tx_end = min(w, tx + half_patch + 1)

        sy_start = max(0, sy - half_patch)
        sy_end = min(h, sy + half_patch + 1)
        sx_start = max(0, sx - half_patch)
        sx_end = min(w, sx + half_patch + 1)

        # Adjust for boundary
        t_patch = image[ty_start:ty_end, tx_start:tx_end]
        s_patch = image[sy_start:sy_end, sx_start:sx_end]
        t_guide = guide[ty_start:ty_end, tx_start:tx_end]
        s_guide = guide[sy_start:sy_end, sx_start:sx_end]
        t_mask = mask[ty_start:ty_end, tx_start:tx_end]

        # Ensure same size
        min_h = min(t_patch.shape[0], s_patch.shape[0])
        min_w = min(t_patch.shape[1], s_patch.shape[1])

        if min_h <= 0 or min_w <= 0:
            return float('inf')

        t_patch = t_patch[:min_h, :min_w]
        s_patch = s_patch[:min_h, :min_w]
        t_guide = t_guide[:min_h, :min_w]
        s_guide = s_guide[:min_h, :min_w]
        t_mask = t_mask[:min_h, :min_w]

        # Only compare known pixels in target patch
        valid = ~t_mask
        if not valid.any():
            return float('inf')

        # Color distance
        color_dist = np.sum((t_patch[valid].astype(float) - s_patch[valid].astype(float)) ** 2)

        # Guidance distance
        if len(t_guide.shape) == 2:
            guide_dist = np.sum((t_guide[valid].astype(float) - s_guide[valid].astype(float)) ** 2)
        else:
            guide_dist = 0

        # Combined distance
        total_dist = (1 - alpha) * color_dist + alpha * guide_dist

        return total_dist / valid.sum()

    @staticmethod
    def _reconstruct_from_nnf(image, mask, nnf, target_coords, patch_size):
        """Reconstruct inpainted image from nearest neighbor field"""
        h, w, c = image.shape
        result = image.copy()
        half_patch = patch_size // 2

        # Vote-based reconstruction
        vote_count = np.zeros((h, w), dtype=np.float32)
        vote_sum = np.zeros((h, w, c), dtype=np.float32)

        for idx, (ty, tx) in enumerate(target_coords):
            sy, sx = nnf[idx]

            # Copy patch
            for dy in range(-half_patch, half_patch + 1):
                for dx in range(-half_patch, half_patch + 1):
                    ty_curr = ty + dy
                    tx_curr = tx + dx
                    sy_curr = sy + dy
                    sx_curr = sx + dx

                    if (0 <= ty_curr < h and 0 <= tx_curr < w and
                        0 <= sy_curr < h and 0 <= sx_curr < w and
                        mask[ty_curr, tx_curr]):
                        # Gaussian weighting
                        weight = np.exp(-(dy**2 + dx**2) / (2 * (half_patch/2)**2))
                        vote_sum[ty_curr, tx_curr] += image[sy_curr, sx_curr] * weight
                        vote_count[ty_curr, tx_curr] += weight

        # Average votes
        mask_3ch = np.stack([mask] * c, axis=-1)
        vote_count_3ch = np.stack([vote_count] * c, axis=-1)
        valid = vote_count_3ch > 0

        result = np.where(valid & mask_3ch,
                          vote_sum / np.maximum(vote_count_3ch, 1e-8),
                          result)

        return result


    @staticmethod
    def inpaint_with_structure_guide(image, mask, edge_threshold=100):
        """
        Convenience method for structure-guided inpainting using edge detection

        Args:
            image: Input image (RGB)
            mask: Binary mask
            edge_threshold: Threshold for Canny edge detection

        Returns:
            Inpainted image
        """
        # Generate structure guide using Canny edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)

        return PatchMatchInpainting.inpaint(image, mask, guide=edges, alpha=0.3)
