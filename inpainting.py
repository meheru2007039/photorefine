import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load image in RGB format"""
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_flare_region(img, threshold=240):
    """
    Detect bright flare regions using thresholding
    Returns a binary mask of the flare area
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Threshold to find very bright regions
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate to capture surrounding affected areas
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

def method1_telea_inpainting(img, mask):
    """
    Method 1: Telea's Inpainting Algorithm
    Fast marching method - good for smaller regions
    """
    result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result

def method2_ns_inpainting(img, mask):
    """
    Method 2: Navier-Stokes Inpainting
    Better for larger regions and preserving structure
    """
    result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    return result

def method3_bilateral_reconstruction(img, mask):
    # Apply bilateral filter to reduce flare while preserving edges
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=100)
    
    # Create inverse mask (areas to keep from original)
    inv_mask = cv2.bitwise_not(mask)
    
    # Normalize masks for blending
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
    inv_mask_3ch = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
    
    # Blend: use filtered image in flare areas, original elsewhere
    result = (bilateral.astype(float) * mask_3ch + 
              img.astype(float) * inv_mask_3ch).astype(np.uint8)
    
    return result

def method4_morphological_reconstruction(img, mask):
    """
    Method 4: Morphological operations + Gaussian blending
    """
    # Close operation to fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Gaussian blur the affected region heavily
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    
    # Create smooth transition mask
    mask_smooth = cv2.GaussianBlur(closed_mask, (7, 7), 0).astype(float) / 255.0
    mask_smooth_3ch = np.stack([mask_smooth] * 3, axis=-1)
    
    # Blend original and blurred based on smooth mask
    result = (img.astype(float) * (1 - mask_smooth_3ch) + 
              blurred.astype(float) * mask_smooth_3ch).astype(np.uint8)
    
    return result

def method5_multiscale_decomposition(img, mask):
    """
    Method 5: Multi-scale decomposition with guided filter
    Separates base and detail, removes flare from base layer
    """
    # Convert to float
    img_float = img.astype(float) / 255.0
    
    # Create base layer using guided filter (approximation with bilateral)
    base = cv2.bilateralFilter(img, d=15, sigmaColor=80, sigmaSpace=80).astype(float) / 255.0
    
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

def visualize_results(img, mask, results, titles):
    """
    Display original, mask, and all results
    """
    n_results = len(results)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Detected mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Detected Flare Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Results
    for i, (result, title) in enumerate(zip(results, titles)):
        axes[i + 2].imshow(result)
        axes[i + 2].set_title(title, fontsize=12, fontweight='bold')
        axes[i + 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load your image
    image_path = "images.png"  # Replace with your image path
    img = load_image(image_path)
    
    # Detect flare region
    mask = detect_flare_region(img, threshold=240)
    
    # Apply different methods
    result1 = method1_telea_inpainting(img, mask)
    result2 = method2_ns_inpainting(img, mask)
    result3 = method3_bilateral_reconstruction(img, mask)
    result4 = method4_morphological_reconstruction(img, mask)
    result5 = method5_multiscale_decomposition(img, mask)
    
    results = [result1, result2, result3, result4, result5]
    titles = [
        'Method 1: Telea Inpainting',
        'Method 2: Navier-Stokes',
        'Method 3: Bilateral Filter',
        'Method 4: Morphological',
        'Method 5: Multiscale'
    ]
    
    # Visualize
    visualize_results(img, mask, results, titles)
    
    # Save the best result (you can choose based on visual inspection)
    cv2.imwrite('result_telea.jpg', cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
    print("Processing complete! Results saved.")