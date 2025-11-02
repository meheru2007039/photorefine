"""
Contrast enhancement and histogram analysis tools
"""

import numpy as np
import cv2
from scipy import ndimage


class ContrastEnhancement:
    """Various contrast enhancement techniques"""

    @staticmethod
    def histogram_equalization(image):
        """
        Apply histogram equalization to enhance contrast

        Args:
            image: Input image (RGB)

        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            result = cv2.equalizeHist(image)

        return result

    @staticmethod
    def adaptive_histogram_equalization(image, clip_limit=2.0, tile_size=8):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        Args:
            image: Input image (RGB)
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization

        Returns:
            Adaptively enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

        if len(image.shape) == 3:
            # Apply to each channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            result = clahe.apply(image)

        return result

    @staticmethod
    def gamma_correction(image, gamma=1.0):
        """
        Apply gamma correction for brightness/contrast adjustment

        Args:
            image: Input image (RGB)
            gamma: Gamma value (< 1 = darker, > 1 = brighter)

        Returns:
            Gamma-corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in range(256)]).astype(np.uint8)

        return cv2.LUT(image, table)

    @staticmethod
    def linear_contrast_stretch(image, percentile_low=2, percentile_high=98):
        """
        Stretch contrast using percentile normalization

        Args:
            image: Input image (RGB)
            percentile_low: Lower percentile for stretching
            percentile_high: Upper percentile for stretching

        Returns:
            Contrast-stretched image
        """
        result = image.copy().astype(np.float32)

        for c in range(image.shape[2] if len(image.shape) == 3 else 1):
            if len(image.shape) == 3:
                channel = image[:, :, c]
            else:
                channel = image

            p_low = np.percentile(channel, percentile_low)
            p_high = np.percentile(channel, percentile_high)

            # Stretch
            stretched = (channel.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
            stretched = np.clip(stretched, 0, 255)

            if len(image.shape) == 3:
                result[:, :, c] = stretched
            else:
                result = stretched

        return result.astype(np.uint8)

    @staticmethod
    def sigmoid_contrast(image, gain=10, cutoff=0.5):
        """
        Apply sigmoid contrast adjustment

        Args:
            image: Input image (RGB)
            gain: Steepness of the sigmoid curve
            cutoff: Center point of the curve (0-1)

        Returns:
            Sigmoid-adjusted image
        """
        normalized = image.astype(np.float32) / 255.0

        # Sigmoid function
        adjusted = 1.0 / (1.0 + np.exp(-gain * (normalized - cutoff)))

        # Normalize back to 0-1 range
        adjusted = (adjusted - adjusted.min()) / (adjusted.max() - adjusted.min())

        return (adjusted * 255).astype(np.uint8)

    @staticmethod
    def local_contrast_enhancement(image, kernel_size=9, strength=1.0):
        """
        Enhance local contrast using unsharp masking

        Args:
            image: Input image (RGB)
            kernel_size: Size of Gaussian kernel for blurring
            strength: Enhancement strength

        Returns:
            Locally enhanced image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        result = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def auto_contrast(image):
        """
        Automatic contrast adjustment by stretching to full range

        Args:
            image: Input image (RGB)

        Returns:
            Auto-contrast adjusted image
        """
        result = image.copy().astype(np.float32)

        for c in range(image.shape[2] if len(image.shape) == 3 else 1):
            if len(image.shape) == 3:
                channel = image[:, :, c]
            else:
                channel = image

            min_val = channel.min()
            max_val = channel.max()

            if max_val > min_val:
                stretched = (channel.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
            else:
                stretched = channel.astype(np.float32)

            if len(image.shape) == 3:
                result[:, :, c] = stretched
            else:
                result = stretched

        return result.astype(np.uint8)


class HistogramAnalysis:
    """Histogram analysis and visualization tools"""

    @staticmethod
    def compute_histogram(image, channel=None):
        """
        Compute histogram for image

        Args:
            image: Input image (RGB or grayscale)
            channel: Specific channel to analyze (0=R/Gray, 1=G, 2=B, None=all)

        Returns:
            Histogram array or list of histograms
        """
        if len(image.shape) == 2:
            # Grayscale
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return hist.flatten()
        else:
            if channel is not None:
                hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
                return hist.flatten()
            else:
                # Compute for all channels
                histograms = []
                for c in range(3):
                    hist = cv2.calcHist([image], [c], None, [256], [0, 256])
                    histograms.append(hist.flatten())
                return histograms

    @staticmethod
    def analyze_histogram(image):
        """
        Analyze histogram and return statistics

        Args:
            image: Input image (RGB)

        Returns:
            Dictionary with histogram statistics
        """
        stats = {}

        if len(image.shape) == 2:
            channels = [image]
            names = ['Gray']
        else:
            channels = [image[:, :, i] for i in range(3)]
            names = ['Red', 'Green', 'Blue']

        for name, channel in zip(names, channels):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()

            stats[name] = {
                'mean': np.mean(channel),
                'std': np.std(channel),
                'min': np.min(channel),
                'max': np.max(channel),
                'median': np.median(channel),
                'mode': np.argmax(hist),  # Most frequent value
                'dynamic_range': np.max(channel) - np.min(channel),
                'histogram': hist
            }

        return stats

    @staticmethod
    def create_histogram_image(image, size=(512, 400)):
        """
        Create a visualization of the histogram

        Args:
            image: Input image (RGB or grayscale)
            size: Size of output histogram image (width, height)

        Returns:
            Histogram visualization as RGB image
        """
        w, h = size
        hist_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        if len(image.shape) == 2:
            # Grayscale histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / hist.max() * (h - 40)  # Scale to fit

            bin_w = w // 256
            for i in range(256):
                x = i * bin_w
                y = h - int(hist[i])
                cv2.line(hist_img, (x, h - 20), (x, y), (0, 0, 0), bin_w)

        else:
            # Color histogram (R, G, B)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB in RGB format
            color_names = ['R', 'G', 'B']

            for idx, color in enumerate(colors):
                hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
                hist = hist.flatten()
                hist = hist / hist.max() * (h - 60)  # Scale to fit

                bin_w = w // 256
                for i in range(256):
                    x = i * bin_w
                    y = h - int(hist[i])
                    cv2.line(hist_img, (x, h - 20), (x, y), color, max(1, bin_w // 2))

            # Add legend
            for idx, (color, name) in enumerate(zip(colors, color_names)):
                cv2.putText(hist_img, name, (10 + idx * 30, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add grid lines
        for i in range(0, h, 50):
            cv2.line(hist_img, (0, i), (w, i), (200, 200, 200), 1)

        return hist_img

    @staticmethod
    def detect_glare_from_histogram(image, brightness_threshold=220, bright_pixel_ratio=0.01):
        """
        Detect potential glare regions by analyzing histogram

        Args:
            image: Input image (RGB)
            brightness_threshold: Threshold for bright pixels
            bright_pixel_ratio: Minimum ratio of bright pixels to consider as glare

        Returns:
            Dictionary with glare detection results
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Count bright pixels
        bright_pixels = np.sum(gray > brightness_threshold)
        total_pixels = gray.size
        ratio = bright_pixels / total_pixels

        # Analyze histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        # Check for peak in bright region
        bright_hist_sum = np.sum(hist[brightness_threshold:])
        total_hist_sum = np.sum(hist)
        bright_hist_ratio = bright_hist_sum / total_hist_sum if total_hist_sum > 0 else 0

        has_glare = ratio > bright_pixel_ratio or bright_hist_ratio > bright_pixel_ratio

        return {
            'has_glare': has_glare,
            'bright_pixel_count': bright_pixels,
            'bright_pixel_ratio': ratio,
            'brightness_threshold': brightness_threshold,
            'recommended_threshold': np.percentile(gray, 98),  # 98th percentile
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray)
        }

    @staticmethod
    def suggest_contrast_adjustment(image):
        """
        Suggest contrast adjustment based on histogram analysis

        Args:
            image: Input image (RGB)

        Returns:
            Dictionary with suggestions
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        dynamic_range = gray.max() - gray.min()
        std_dev = np.std(gray)
        mean_val = np.mean(gray)

        suggestions = []

        if dynamic_range < 150:
            suggestions.append("Low dynamic range - consider contrast stretching")

        if std_dev < 30:
            suggestions.append("Low contrast - apply histogram equalization")

        if mean_val < 80:
            suggestions.append("Dark image - apply gamma correction (gamma > 1)")
        elif mean_val > 170:
            suggestions.append("Bright image - apply gamma correction (gamma < 1)")

        if len(suggestions) == 0:
            suggestions.append("Image has good contrast distribution")

        return {
            'dynamic_range': dynamic_range,
            'std_dev': std_dev,
            'mean': mean_val,
            'suggestions': suggestions
        }
