
import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import distance_transform_edt
import copy
import cv2


class ImageProcessor:
    
    @staticmethod
    def rgb_to_grayscale(image):
        """Convert RGB to grayscale using OpenCV"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    @staticmethod
    def rgb_to_hsv(rgb):
        """Convert RGB to HSV color space using OpenCV"""
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    @staticmethod
    def hsv_to_rgb(hsv):
        """Convert HSV to RGB color space using OpenCV"""
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    @staticmethod
    def create_gaussian_kernel(sigma, ksize=None):
        """Create 2D Gaussian kernel"""
        if ksize is None:
            ksize = int(6 * sigma) + 1
        if ksize % 2 == 0:
            ksize += 1
        
        center = ksize // 2
        x = np.arange(-center, center + 1)
        y = np.arange(-center, center + 1)
        X, Y = np.meshgrid(x, y)
        
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        kernel = kernel / (2 * np.pi * sigma**2)
        kernel = kernel / kernel.sum()
        
        return kernel.astype(np.float32)
    
    @staticmethod
    def convolve2d(image, kernel, padding='reflect'):
        """2D convolution for single channel"""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        if padding == 'reflect':
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        else:
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        output = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kh, j:j+kw]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    @staticmethod
    def gaussian_blur(image, sigma=1.0, ksize=None):
        """Apply Gaussian blur to image"""
        kernel = ImageProcessor.create_gaussian_kernel(sigma, ksize)
        
        if len(image.shape) == 2:
            return ImageProcessor.convolve2d(image, kernel).astype(np.uint8)
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[..., c] = ImageProcessor.convolve2d(image[..., c], kernel)
            return result.astype(np.uint8)
    
    @staticmethod
    def threshold_binary(image, thresh, max_val=255):
        """Simple binary thresholding"""
        result = np.zeros_like(image)
        result[image > thresh] = max_val
        return result
    
    @staticmethod
    def adaptive_threshold(image, block_size=11, c=2):
        """Adaptive thresholding using local mean"""
        if block_size % 2 == 0:
            block_size += 1
        
        # local mean using box filter
        pad = block_size // 2
        padded = np.pad(image, pad, mode='reflect')
        
        result = np.zeros_like(image, dtype=np.uint8)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                local_region = padded[i:i+block_size, j:j+block_size]
                local_mean = np.mean(local_region)
                threshold = local_mean - c
                result[i, j] = 255 if image[i, j] < threshold else 0
        
        return result
    
    @staticmethod
    def sobel_gradient(image):
        """Calculate image gradient using Sobel operator"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        if len(image.shape) == 3:
            image = ImageProcessor.rgb_to_grayscale(image)
        
        gx = ImageProcessor.convolve2d(image.astype(np.float32), sobel_x)
        gy = ImageProcessor.convolve2d(image.astype(np.float32), sobel_y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        return magnitude, direction
    
    @staticmethod
    def non_max_suppression(magnitude, direction):
        """Non-maximum suppression for edge detection"""
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # Convert angle to degrees
        angle = np.rad2deg(direction) % 180
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                a = angle[i, j]
                
                # Determine neighbors based on gradient direction
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= a < 67.5:
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif 67.5 <= a < 112.5:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 112.5 <= a < 157.5
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                
                if magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = magnitude[i, j]
        
        return suppressed
    
    @staticmethod
    def hysteresis_threshold(image, low_thresh, high_thresh):
        """Hysteresis thresholding"""
        strong = image >= high_thresh
        weak = (image >= low_thresh) & (image < high_thresh)
        
        result = np.zeros_like(image, dtype=np.uint8)
        result[strong] = 255
        
        # Edge tracking by hysteresis
        for _ in range(5):
            dilated = ndimage.binary_dilation(result > 0)
            result[weak & dilated] = 255
        
        return result
    
    @staticmethod
    def canny_edge_detection(image, low_thresh=50, high_thresh=150, sigma=1.0):
        """Canny edge detection algorithm"""
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_grayscale(image)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = ImageProcessor.gaussian_blur(gray, sigma=sigma)
        
        # Calculate gradients
        magnitude, direction = ImageProcessor.sobel_gradient(blurred)
        
        # Non-maximum suppression
        suppressed = ImageProcessor.non_max_suppression(magnitude, direction)
        
        # Double thresholding and edge tracking
        edges = ImageProcessor.hysteresis_threshold(suppressed, low_thresh, high_thresh)
        
        return edges
    
    @staticmethod
    def dilate(image, kernel_size=3, iterations=1):
        """Morphological dilation using OpenCV"""
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        result = cv2.dilate(image, kernel, iterations=iterations)
        return result
    
    @staticmethod
    def erode(image, kernel_size=3, iterations=1):
        """Morphological erosion using OpenCV"""
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        result = cv2.erode(image, kernel, iterations=iterations)
        return result
    
    @staticmethod
    def morphology_open(image, kernel_size=3, iterations=1):
        """erosion followed by dilation"""
        result = ImageProcessor.erode(image, kernel_size, iterations)
        result = ImageProcessor.dilate(result, kernel_size, iterations)
        return result
    
    @staticmethod
    def morphology_close(image, kernel_size=3, iterations=1):
        """dilation followed by erosion"""
        result = ImageProcessor.dilate(image, kernel_size, iterations)
        result = ImageProcessor.erode(result, kernel_size, iterations)
        return result
    
    @staticmethod
    def find_contours(binary_image):
        """Find contours in binary image using boundary following"""
        labeled, num_features = ndimage.label(binary_image > 0)
        contours = []
        
        for label_id in range(1, num_features + 1):
            mask = (labeled == label_id).astype(np.uint8)
            
            # Find boundary points
            eroded = ImageProcessor.erode(mask, kernel_size=3, iterations=1)
            boundary = mask - eroded
            
            # Get coordinates of boundary points
            points = np.column_stack(np.where(boundary > 0))
            
            if len(points) > 0:
                # Convert to (x, y) format and sort
                contour = points[:, [1, 0]]  # Swap to x, y
                contours.append(contour)
        
        return contours
    
    @staticmethod
    def contour_area(contour):
        """Calculate contour area using shoelace formula"""
        if len(contour) < 3:
            return 0
        
        x = contour[:, 0]
        y = contour[:, 1]
        
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
    
    @staticmethod
    def contour_perimeter(contour):
        """Calculate contour perimeter"""
        if len(contour) < 2:
            return 0
        
        distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
        # Add distance from last to first point
        distances = np.append(distances, np.linalg.norm(contour[-1] - contour[0]))
        
        return np.sum(distances)
    
    @staticmethod
    def bounding_rect(contour):
        """Calculate bounding rectangle"""
        x_min = np.min(contour[:, 0])
        x_max = np.max(contour[:, 0])
        y_min = np.min(contour[:, 1])
        y_max = np.max(contour[:, 1])
        
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
    
    @staticmethod
    def contour_moments(contour):
        """Calculate contour moments"""
        if len(contour) < 1:
            return {'m00': 0, 'm10': 0, 'm01': 0}
        
        x = contour[:, 0]
        y = contour[:, 1]
        
        m00 = len(contour)
        m10 = np.sum(x)
        m01 = np.sum(y)
        
        return {'m00': m00, 'm10': m10, 'm01': m01}
    
    @staticmethod
    def kmeans_segmentation(image, n_clusters=3, max_iter=100):
        """K-means clustering for image segmentation"""
        pixels = image.reshape((-1, 3)).astype(np.float32)
        
        # Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(len(pixels), n_clusters, replace=False)
        centroids = pixels[indices]
        
        for _ in range(max_iter):
            # Assign pixels to nearest centroid
            distances = np.sqrt(((pixels[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([pixels[labels == k].mean(axis=0) 
                                     for k in range(n_clusters)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        # Create segmented image
        segmented = centroids[labels].reshape(image.shape)
        
        # Create mask for brightest cluster
        brightness = np.mean(centroids, axis=1)
        brightest_label = np.argmax(brightness)
        mask = (labels.reshape(image.shape[:2]) == brightest_label).astype(np.uint8) * 255
        
        return mask, segmented.astype(np.uint8)
    
    @staticmethod
    def inpaint_telea(image, mask, radius=3):
        """Simple inpainting using diffusion-based method"""
        mask_binary = (mask > 0).astype(bool)
        
        mask_dilated = ImageProcessor.dilate(mask_binary.astype(np.uint8) * 255, 
                                                 kernel_size=3, iterations=1) > 0
        
        result = image.copy().astype(np.float32)

        for iteration in range(radius * 10):
            # For each channel
            for c in range(image.shape[2] if len(image.shape) == 3 else 1):
                if len(image.shape) == 3:
                    channel = result[..., c]
                else:
                    channel = result
                
                # Average filter on the masked region
                filtered = ndimage.uniform_filter(channel, size=5, mode='reflect')
                
                # Update only masked pixels
                if len(image.shape) == 3:
                    result[..., c] = np.where(mask_dilated, filtered, result[..., c])
                else:
                    result = np.where(mask_dilated, filtered, result)
        
        return result.astype(np.uint8)


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