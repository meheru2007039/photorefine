import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import distance_transform_edt
import copy
import cv2

class BlobRemovalProcessor:
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.working_image = None
        self.debug_images = {}
        self.detected_mask = None
        self.current_mask = None
        self.processing_history = []
        self.current_step = -1
        self.processor = ImageProcessor()
        
    def load_image(self, image_path):
        
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.processed_image = self.original_image.copy()
        self.working_image = self.original_image.copy()
        self.processing_history = []
        self.current_step = -1
        return self.original_image
    
    def reset_to_original(self):
        if self.original_image is not None:
            self.working_image = self.original_image.copy()
            self.processed_image = self.original_image.copy()
            self.processing_history = []
            self.current_step = -1
            self.current_mask = None
            return True
        return False
    
    def reset_to_last_inpainted(self):
        """Reset to last inpainted result"""
        if self.processed_image is not None:
            self.working_image = self.processed_image.copy()
            return True
        return False
    
    def get_working_image(self):
        """Get the current working image"""
        return self.working_image if self.working_image is not None else self.original_image
    
    def add_to_history(self, params, mask, result):
        """Add processing step to history"""
        self.processing_history = self.processing_history[:self.current_step + 1]
        
        self.processing_history.append({
            'params': copy.deepcopy(params),
            'mask': mask.copy() if mask is not None else None,
            'result': result.copy() if result is not None else None,
            'timestamp': len(self.processing_history)
        })
        self.current_step = len(self.processing_history) - 1
    
    def undo_step(self):
        """Go back one step in history"""
        if self.current_step > 0:
            self.current_step -= 1
            step = self.processing_history[self.current_step]
            self.working_image = step['result'].copy()
            self.processed_image = step['result'].copy()
            self.current_mask = step['mask']
            return True, self.current_step, len(self.processing_history)
        elif self.current_step == 0:
            self.reset_to_original()
            return True, -1, len(self.processing_history)
        return False, self.current_step, len(self.processing_history)
    
    def redo_step(self):
        """Go forward one step in history"""
        if self.current_step < len(self.processing_history) - 1:
            self.current_step += 1
            step = self.processing_history[self.current_step]
            self.working_image = step['result'].copy()
            self.processed_image = step['result'].copy()
            self.current_mask = step['mask']
            return True, self.current_step, len(self.processing_history)
        return False, self.current_step, len(self.processing_history)
    
    def get_history_info(self):
        """Get information about processing history"""
        return {
            'total_steps': len(self.processing_history),
            'current_step': self.current_step,
            'can_undo': self.current_step >= 0,
            'can_redo': self.current_step < len(self.processing_history) - 1
        }
    
    def create_color_mask_hsv(self, image, h_min, h_max, s_min, s_max, v_min, v_max):
        """Create mask based on HSV thresholds"""
        hsv = self.processor.rgb_to_hsv(image)
        
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        
        mask = ((h >= h_min) & (h <= h_max) & 
                (s >= s_min) & (s <= s_max) & 
                (v >= v_min) & (v <= v_max)).astype(np.uint8) * 255
        
        self.debug_images['hsv_mask'] = mask
        return mask
    
    def create_brightness_mask(self, image, threshold):
        """Create mask for bright regions"""
        gray = self.processor.rgb_to_grayscale(image)
        mask = self.processor.threshold_binary(gray, threshold)
        self.debug_images['brightness_mask'] = mask
        return mask
    
    def create_saturation_mask(self, image, threshold):
        """Create mask based on saturation threshold"""
        hsv = self.processor.rgb_to_hsv(image)
        s = hsv[..., 1]
        mask = self.processor.threshold_binary(s, threshold)
        self.debug_images['saturation_mask'] = mask
        return mask
    
    def create_edge_based_mask(self, image, canny_low, canny_high):
        """Create mask using Canny edge detection"""
        edges = self.processor.canny_edge_detection(image, canny_low, canny_high)
        
        # Dilate edges
        edges_dilated = self.processor.dilate(edges, kernel_size=5, iterations=2)
        
        # Find contours and fill
        contours = self.processor.find_contours(edges_dilated)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for contour in contours:
            # Create mask for this contour
            contour_mask = np.zeros_like(mask)
            for point in contour:
                x, y = int(point[0]), int(point[1])
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    contour_mask[y, x] = 255
            
            # Fill the contour
            filled = ndimage.binary_fill_holes(contour_mask).astype(np.uint8) * 255
            mask = np.maximum(mask, filled)
        
        self.debug_images['edge_mask'] = edges
        self.debug_images['edge_regions'] = mask
        return mask
    
    def create_adaptive_threshold_mask(self, image, block_size, c_value):
        """Create mask using adaptive thresholding"""
        gray = self.processor.rgb_to_grayscale(image)
        mask = self.processor.adaptive_threshold(gray, block_size, c_value)
        self.debug_images['adaptive_mask'] = mask
        return mask
    
    def morphological_operations(self, mask, operation, kernel_size, iterations=1):
        """Apply morphological operations to mask with iterations"""
        if operation == 'open':
            result = self.processor.morphology_open(mask, kernel_size, iterations)
        elif operation == 'close':
            result = self.processor.morphology_close(mask, kernel_size, iterations)
        elif operation == 'dilate':
            result = self.processor.dilate(mask, kernel_size, iterations)
        elif operation == 'erode':
            result = self.processor.erode(mask, kernel_size, iterations)
        else:
            result = mask
        
        return result
    
    def filter_contours_by_area(self, mask, min_area, max_area):
        """Filter mask by contour area"""
        contours = self.processor.find_contours(mask)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = self.processor.contour_area(contour)
            if min_area <= area <= max_area:
                # Draw contour
                for point in contour:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                        filtered_mask[y, x] = 255
                
                # Fill
                filled = ndimage.binary_fill_holes(filtered_mask).astype(np.uint8) * 255
                filtered_mask = filled
        
        self.debug_images['filtered_contours'] = filtered_mask
        return filtered_mask
    
    def cluster_based_segmentation(self, image, n_clusters):
        """Segment image using k-means clustering"""
        mask, segmented = self.processor.kmeans_segmentation(image, n_clusters)
        self.debug_images['kmeans_segmented'] = segmented
        return mask
    
    def detect_blobs_with_contours(self, mask):
        """Detect and analyze blobs using contour analysis"""
        contours = self.processor.find_contours(mask)
        
        blob_info = []
        debug_image = np.stack([mask, mask, mask], axis=2)
        
        for contour in contours:
            area = self.processor.contour_area(contour)
            if area > 50:
                perimeter = self.processor.contour_perimeter(contour)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                x, y, w, h = self.processor.bounding_rect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                moments = self.processor.contour_moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = 0, 0
                
                blob_info.append({
                    'contour': contour,
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'center': (cx, cy)
                })
        
        self.debug_images['contour_analysis'] = debug_image
        return blob_info
    
    def inpaint_regions(self, image, mask, inpaint_radius):
        """Inpaint masked regions"""
        mask_binary = (mask > 0).astype(np.uint8) * 255
        mask_dilated = self.processor.dilate(mask_binary, kernel_size=3, iterations=1)
        
        result = self.processor.inpaint_telea(image, mask_dilated, radius=inpaint_radius)
        
        self.debug_images['inpainted'] = result
        return result
    
    def process_with_params(self, params, use_working_image=True, add_to_history=True):
        """Process image with given parameters"""
        if self.original_image is None:
            return None, None
        
        if use_working_image and self.working_image is not None:
            image = self.working_image.copy()
        else:
            image = self.original_image.copy()
        
        method = params['detection_method']
        
        # Create mask based on selected method
        if method == 'hsv':
            mask = self.create_color_mask_hsv(image, params['h_min'], params['h_max'],
                                             params['s_min'], params['s_max'],
                                             params['v_min'], params['v_max'])
        elif method == 'brightness':
            mask = self.create_brightness_mask(image, params['brightness_threshold'])
        elif method == 'saturation':
            mask = self.create_saturation_mask(image, params['saturation_threshold'])
        elif method == 'edge':
            mask = self.create_edge_based_mask(image, params['canny_low'], params['canny_high'])
        elif method == 'adaptive':
            mask = self.create_adaptive_threshold_mask(image, params['block_size'], 
                                                       params['c_value'])
        elif method == 'kmeans':
            mask = self.cluster_based_segmentation(image, params['n_clusters'])
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply morphological operations
        if params['morph_operation'] != 'none':
            mask = self.morphological_operations(mask, params['morph_operation'], 
                                                 params['kernel_size'])
        
        # Filter by area
        mask = self.filter_contours_by_area(mask, params['min_area'], params['max_area'])
        
        # Detect blobs
        blob_info = self.detect_blobs_with_contours(mask)
        
        # Apply inpainting if enabled
        if params['enable_inpaint']:
            result = self.inpaint_regions(image, mask, params['inpaint_radius'])
        else:
            result = image
        
        # Update current state
        self.current_mask = mask
        self.processed_image = result
        self.working_image = result.copy()
        
        # Add to history if requested
        if add_to_history and params['enable_inpaint']:
            self.add_to_history(params, mask, result)
        
        return result, mask