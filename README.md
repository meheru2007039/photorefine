# PhotoRefine

**Advanced Image Processing Tool for Glare and Reflection Removal**

PhotoRefine is a powerful, feature-rich application for detecting and removing glare, reflections, and unwanted bright spots from images using state-of-the-art image processing techniques.

## Features

### Detection Methods
- **HSV Color Space Filtering** - Precise color-based detection
- **Brightness Thresholding** - Simple intensity-based detection
- **Saturation-based Detection** - Isolate oversaturated regions
- **Edge Detection (Canny)** - Boundary-based detection
- **Adaptive Thresholding** - Local contrast-based detection
- **K-Means Clustering** - Automatic color segmentation
- **🆕 Watershed Segmentation** - Advanced color segmentation with marker-based region growing
- **🆕 DFT (Fourier) Filtering** - Frequency domain filtering for uniform reflections

### Inpainting Algorithms
- **Telea Method** - Fast marching inpainting (fast, good for small regions)
- **Navier-Stokes** - Fluid dynamics-based inpainting (better for larger areas)
- **Bilateral Filter** - Edge-preserving reconstruction
- **Morphological Reconstruction** - Structure-preserving filling
- **Multiscale Decomposition** - Base+detail layer separation
- **🆕 PatchMatch** - State-of-the-art exemplar-based inpainting with guidance support

### Advanced Features
- **Multi-pass Processing** - Chain multiple filters for complex cases
- **Undo/Redo History** - Full editing history with step-by-step navigation
- **Morphological Operations** - Open, close, dilate, erode with iterations
- **Area Filtering** - Remove artifacts by size constraints
- **Real-time Preview** - Interactive parameter adjustment with live mask preview
- **Structure Guidance** - Guide inpainting with edge/structure information

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the GUI Application

```bash
python main.py
```

Or directly:
```bash
python -m photorefine.gui.main_window
```

### Using as a Library

```python
from photorefine import BlobRemovalProcessor
from photorefine.filters import WatershedSegmentation, DFTFilter
from photorefine.inpainting import PatchMatchInpainting

# Load and process an image
processor = BlobRemovalProcessor()
processor.load_image("path/to/image.jpg")

# Example 1: Watershed segmentation
params = {
    'detection_method': 'watershed',
    'watershed_markers': 5,
    'watershed_compactness': 0.001,
    'enable_inpaint': True,
    'inpaint_method': 'patchmatch',
    'patchmatch_patch_size': 7,
    'patchmatch_iterations': 5,
    'min_area': 100,
    'max_area': 50000
}

result, mask = processor.process_with_params(params)

# Example 2: DFT filtering for uniform reflections
from photorefine.filters import DFTFilter
import cv2

image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Remove uniform reflection patterns
result = DFTFilter.remove_uniform_reflection(
    image,
    filter_type='notch',
    radius=10,
    auto_detect=True
)

# Example 3: PatchMatch inpainting with structure guidance
from photorefine.inpainting import PatchMatchInpainting

# With automatic edge guidance
result = PatchMatchInpainting.inpaint_with_structure_guide(
    image,
    mask,
    edge_threshold=100
)

# With custom guidance
result = PatchMatchInpainting.inpaint(
    image,
    mask,
    guide=custom_edge_map,
    patch_size=7,
    iterations=5,
    alpha=0.5  # 0=no guidance, 1=full guidance
)
```

## Project Structure

```
photorefine/
├── photorefine/
│   ├── __init__.py           # Package initialization
│   ├── core/                 # Core processing modules
│   │   ├── __init__.py
│   │   ├── image_processor.py    # Basic image operations
│   │   └── blob_processor.py     # Blob removal processor
│   ├── filters/              # Advanced filtering algorithms
│   │   ├── __init__.py
│   │   ├── color_segmentation.py # Watershed segmentation
│   │   └── fourier.py            # DFT-based filters
│   ├── inpainting/          # Inpainting algorithms
│   │   ├── __init__.py
│   │   ├── basic.py              # Basic inpainting methods
│   │   └── patchmatch.py         # PatchMatch algorithm
│   └── gui/                 # Graphical user interface
│       ├── __init__.py
│       └── main_window.py        # Main GUI application
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Algorithm Details

### Watershed Segmentation
Implements marker-based watershed algorithm for color segmentation:
- Automatically detects bright/saturated regions
- Uses SLIC superpixels for initial segmentation
- Applies watershed transform for precise boundaries
- Configurable markers and compactness parameters

### DFT (Fourier) Filtering
Frequency domain filtering for periodic patterns:
- **Notch Filter**: Removes specific frequencies (auto-detects bright spots)
- **High-pass Filter**: Removes low-frequency illumination changes
- **Band-reject Filter**: Removes specific frequency bands
- **Adaptive Filter**: Automatically identifies anomalous frequencies
- **Homomorphic Filter**: Illumination normalization

### PatchMatch Inpainting
Exemplar-based inpainting with approximate nearest neighbor search:
- Multi-scale pyramid for better results
- Structure-guided matching using edge maps
- Iterative refinement with propagation and random search
- Configurable patch size, iterations, and guidance weight
- Vote-based reconstruction with Gaussian weighting

## Tips for Best Results

1. **For uniform bright spots (glare):**
   - Use Brightness or HSV detection
   - Try DFT filter with notch or adaptive mode
   - Use Telea or NS inpainting for fast results

2. **For colored reflections:**
   - Use HSV or Watershed detection
   - Adjust hue/saturation ranges
   - Use PatchMatch inpainting for complex textures

3. **For periodic patterns (lens flare):**
   - Use DFT filter with notch mode
   - Enable auto-detect for automatic frequency identification
   - Combine with morphological operations

4. **For complex cases:**
   - Enable "Process from Current" mode
   - Apply multiple filters in sequence:
     1. DFT filter to remove uniform components
     2. Watershed to detect remaining spots
     3. PatchMatch to fill with structure guidance
   - Use Undo/Redo to experiment

5. **PatchMatch settings:**
   - Larger patch size (9-11): Better structure preservation
   - Smaller patch size (5-7): Better detail matching
   - Higher guidance (α=0.6-0.8): Follow edges more closely
   - Lower guidance (α=0.2-0.4): More creative filling

## Performance Notes

- **PatchMatch** is computationally intensive; processing may take 10-30 seconds depending on image size and parameters
- Use **preview mode** to adjust detection parameters before applying expensive inpainting
- Consider **reducing image size** for faster experimentation
- **Multi-scale processing** helps PatchMatch run faster while maintaining quality

## Troubleshooting

**Issue**: Mask detects too much
- Increase minimum area threshold
- Adjust HSV/brightness ranges to be more selective
- Use morphological "open" operation to remove small artifacts

**Issue**: Mask misses some regions
- Decrease brightness threshold
- Increase HSV ranges
- Try watershed segmentation with more markers
- Use morphological "close" operation to connect nearby regions

**Issue**: Inpainting creates artifacts
- Try different inpainting methods
- Increase inpaint radius
- For PatchMatch: adjust patch size and guidance weight
- Use multi-pass approach: first rough inpaint, then refine

**Issue**: DFT filter not working
- Ensure reflection has uniform frequency signature
- Try different filter types (notch, adaptive, highpass)
- Adjust cutoff frequency and radius parameters
- Visualize frequency spectrum to identify patterns

## References

- **PatchMatch Algorithm**: Barnes, C., Shechtman, E., Finkelstein, A., & Goldman, D. B. (2009). PatchMatch: A randomized correspondence algorithm for structural image editing. *ACM SIGGRAPH 2009*.

- **Watershed Segmentation**: Vincent, L., & Soille, P. (1991). Watersheds in digital spaces: an efficient algorithm based on immersion simulations. *IEEE TPAMI*.

- **Telea Inpainting**: Telea, A. (2004). An image inpainting technique based on the fast marching method. *Journal of graphics tools*.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional detection methods (deep learning-based)
- GPU acceleration for PatchMatch
- Batch processing support
- Additional inpainting algorithms
- Plugin system for custom filters

## Contact

For questions, issues, or suggestions, please open an issue on the project repository.

---

**Made with ❤️ for better image processing**