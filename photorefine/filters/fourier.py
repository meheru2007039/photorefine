"""
DFT (Discrete Fourier Transform) based filters for uniform reflection/glare removal
"""

import numpy as np
import cv2


class DFTFilter:
    """
    Fourier transform-based filtering for removing uniform reflections and patterns
    Effective when reflection has a distinct frequency signature
    """

    @staticmethod
    def remove_uniform_reflection(image, filter_type='notch', **kwargs):
        """
        Remove uniform reflection using DFT filtering

        Args:
            image: Input image (RGB)
            filter_type: Type of filter ('notch', 'highpass', 'bandreject', 'adaptive')
            **kwargs: Additional parameters for specific filter types

        Returns:
            Filtered image with reflection removed
        """
        if filter_type == 'notch':
            return DFTFilter.notch_filter(image, **kwargs)
        elif filter_type == 'highpass':
            return DFTFilter.highpass_filter(image, **kwargs)
        elif filter_type == 'bandreject':
            return DFTFilter.bandreject_filter(image, **kwargs)
        elif filter_type == 'adaptive':
            return DFTFilter.adaptive_frequency_filter(image, **kwargs)
        else:
            return DFTFilter.notch_filter(image, **kwargs)

    @staticmethod
    def notch_filter(image, centers=None, radius=10, auto_detect=True):
        """
        Notch filter to remove specific frequencies (useful for uniform patterns)

        Args:
            image: Input image (RGB)
            centers: List of (u, v) frequency coordinates to notch, or None for auto-detect
            radius: Radius of notch filter
            auto_detect: Automatically detect bright spots in frequency domain

        Returns:
            Filtered image
        """
        result = np.zeros_like(image)

        for c in range(image.shape[2]):
            channel = image[:, :, c]

            # Compute DFT
            dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            # Get magnitude spectrum
            magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

            # Create notch filter
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2

            # Auto-detect bright spots if requested
            if auto_detect and centers is None:
                centers = DFTFilter._detect_frequency_peaks(magnitude, crow, ccol, radius)

            # Create filter mask
            mask = np.ones((rows, cols, 2), dtype=np.float32)

            if centers is not None:
                for (u, v) in centers:
                    # Create notch at (u, v) and (-u, -v) for symmetry
                    y, x = np.ogrid[:rows, :cols]

                    # Notch at (u, v)
                    dist1 = np.sqrt((x - ccol - u)**2 + (y - crow - v)**2)
                    # Notch at symmetric position
                    dist2 = np.sqrt((x - ccol + u)**2 + (y - crow + v)**2)

                    # Gaussian notch
                    notch = np.exp(-((dist1**2) / (2 * radius**2)))
                    notch += np.exp(-((dist2**2) / (2 * radius**2)))
                    notch = 1 - np.clip(notch, 0, 1)

                    mask[:, :, 0] *= notch
                    mask[:, :, 1] *= notch

            # Apply filter
            filtered_dft = dft_shift * mask

            # Inverse DFT
            idft_shift = np.fft.ifftshift(filtered_dft)
            idft = cv2.idft(idft_shift)
            filtered = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

            # Normalize
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
            result[:, :, c] = filtered

        return result.astype(np.uint8)

    @staticmethod
    def highpass_filter(image, cutoff_freq=30):
        """
        High-pass filter to remove low-frequency components (uniform illumination)

        Args:
            image: Input image (RGB)
            cutoff_freq: Cutoff frequency (radius in frequency domain)

        Returns:
            High-pass filtered image
        """
        result = np.zeros_like(image)

        for c in range(image.shape[2]):
            channel = image[:, :, c]

            # Compute DFT
            dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            # Create high-pass filter mask
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2

            # Create meshgrid
            y, x = np.ogrid[:rows, :cols]
            dist = np.sqrt((x - ccol)**2 + (y - crow)**2)

            # Gaussian high-pass filter
            mask = 1 - np.exp(-(dist**2) / (2 * cutoff_freq**2))
            mask_3d = np.stack([mask, mask], axis=-1)

            # Apply filter
            filtered_dft = dft_shift * mask_3d

            # Inverse DFT
            idft_shift = np.fft.ifftshift(filtered_dft)
            idft = cv2.idft(idft_shift)
            filtered = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

            # Normalize
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
            result[:, :, c] = filtered

        return result.astype(np.uint8)

    @staticmethod
    def bandreject_filter(image, center_freq=50, bandwidth=20):
        """
        Band-reject filter to remove specific frequency bands

        Args:
            image: Input image (RGB)
            center_freq: Center frequency to reject
            bandwidth: Bandwidth of rejection

        Returns:
            Band-reject filtered image
        """
        result = np.zeros_like(image)

        for c in range(image.shape[2]):
            channel = image[:, :, c]

            # Compute DFT
            dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            # Create band-reject filter
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2

            y, x = np.ogrid[:rows, :cols]
            dist = np.sqrt((x - ccol)**2 + (y - crow)**2)

            # Gaussian band-reject
            mask = 1 - np.exp(-((dist - center_freq)**2) / (2 * bandwidth**2))
            mask_3d = np.stack([mask, mask], axis=-1)

            # Apply filter
            filtered_dft = dft_shift * mask_3d

            # Inverse DFT
            idft_shift = np.fft.ifftshift(filtered_dft)
            idft = cv2.idft(idft_shift)
            filtered = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

            # Normalize
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
            result[:, :, c] = filtered

        return result.astype(np.uint8)

    @staticmethod
    def adaptive_frequency_filter(image, percentile=95, radius=15):
        """
        Adaptive filter that automatically identifies and removes anomalous frequencies

        Args:
            image: Input image (RGB)
            percentile: Percentile threshold for detecting bright frequency components
            radius: Radius for notch filter

        Returns:
            Adaptively filtered image
        """
        result = np.zeros_like(image)

        for c in range(image.shape[2]):
            channel = image[:, :, c]

            # Compute DFT
            dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            # Get magnitude spectrum
            magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

            # Log scale for better visualization
            magnitude_log = np.log(magnitude + 1)

            # Find threshold
            threshold = np.percentile(magnitude_log, percentile)

            # Create adaptive mask
            rows, cols = channel.shape
            mask = np.ones((rows, cols, 2), dtype=np.float32)

            # Suppress frequencies above threshold
            high_freq_mask = magnitude_log > threshold
            crow, ccol = rows // 2, cols // 2

            # Exclude DC component
            high_freq_mask[crow-5:crow+5, ccol-5:ccol+5] = False

            # Create smooth suppression
            y, x = np.ogrid[:rows, :cols]
            for i in range(rows):
                for j in range(cols):
                    if high_freq_mask[i, j]:
                        dist = np.sqrt((x - j)**2 + (y - i)**2)
                        suppression = np.exp(-(dist**2) / (2 * radius**2))
                        suppression = 1 - np.clip(suppression, 0, 0.8)
                        mask[i, j, 0] *= suppression[i, j]
                        mask[i, j, 1] *= suppression[i, j]

            # Apply filter
            filtered_dft = dft_shift * mask

            # Inverse DFT
            idft_shift = np.fft.ifftshift(filtered_dft)
            idft = cv2.idft(idft_shift)
            filtered = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

            # Normalize
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
            result[:, :, c] = filtered

        return result.astype(np.uint8)

    @staticmethod
    def _detect_frequency_peaks(magnitude, crow, ccol, min_dist=20, n_peaks=5):
        """
        Detect bright peaks in frequency domain (excluding DC component)

        Args:
            magnitude: Magnitude spectrum
            crow, ccol: Center coordinates
            min_dist: Minimum distance from center to consider
            n_peaks: Number of peaks to detect

        Returns:
            List of (u, v) frequency coordinates
        """
        # Log scale
        magnitude_log = np.log(magnitude + 1)

        # Mask out DC component and nearby region
        masked = magnitude_log.copy()
        masked[crow-min_dist:crow+min_dist, ccol-min_dist:ccol+min_dist] = 0

        # Find peaks
        peaks = []
        temp = masked.copy()

        for _ in range(n_peaks):
            max_idx = np.unravel_index(np.argmax(temp), temp.shape)
            if temp[max_idx] > 0:
                u = max_idx[1] - ccol
                v = max_idx[0] - crow
                peaks.append((u, v))

                # Suppress nearby region
                y, x = max_idx
                temp[max(0, y-10):min(temp.shape[0], y+10),
                     max(0, x-10):min(temp.shape[1], x+10)] = 0

        return peaks if len(peaks) > 0 else None

    @staticmethod
    def visualize_frequency_spectrum(image, channel=0):
        """
        Visualize the frequency spectrum of an image

        Args:
            image: Input image (RGB)
            channel: Channel to visualize (0, 1, or 2)

        Returns:
            Magnitude spectrum visualization
        """
        channel_data = image[:, :, channel]

        # Compute DFT
        dft = cv2.dft(np.float32(channel_data), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Magnitude spectrum
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Log scale for visualization
        magnitude_log = np.log(magnitude + 1)

        # Normalize for display
        magnitude_display = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)

        return magnitude_display.astype(np.uint8)

    @staticmethod
    def homomorphic_filter(image, gamma_h=2.0, gamma_l=0.5, cutoff=30):
        """
        Homomorphic filtering for illumination correction

        Args:
            image: Input image (RGB)
            gamma_h: High frequency gain
            gamma_l: Low frequency gain
            cutoff: Cutoff frequency

        Returns:
            Illumination-corrected image
        """
        result = np.zeros_like(image, dtype=np.float32)

        for c in range(image.shape[2]):
            channel = image[:, :, c].astype(np.float32) + 1

            # Log transform
            log_channel = np.log(channel)

            # DFT
            dft = cv2.dft(log_channel, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            # Create homomorphic filter
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2

            y, x = np.ogrid[:rows, :cols]
            dist = np.sqrt((x - ccol)**2 + (y - crow)**2)

            # High-frequency emphasis filter
            H = (gamma_h - gamma_l) * (1 - np.exp(-(dist**2) / (2 * cutoff**2))) + gamma_l
            H_3d = np.stack([H, H], axis=-1)

            # Apply filter
            filtered_dft = dft_shift * H_3d

            # Inverse DFT
            idft_shift = np.fft.ifftshift(filtered_dft)
            idft = cv2.idft(idft_shift)
            filtered_log = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

            # Exponential to get back to image domain
            filtered = np.exp(filtered_log) - 1

            result[:, :, c] = filtered

        # Normalize
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
