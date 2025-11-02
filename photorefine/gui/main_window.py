"""
PhotoRefine: Complete GUI with all features properly exposed
- Contrast enhancement and histogram analysis
- All inpainting methods (Telea, NS, Bilateral, Morphological, Multiscale, PatchMatch)
- DFT filtering
- Watershed segmentation
- Multi-pass filtering support
"""

import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import threading
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.blob_processor import BlobRemovalProcessor
from ..filters.color_segmentation import WatershedSegmentation
from ..filters.fourier import DFTFilter
from ..filters.contrast import ContrastEnhancement, HistogramAnalysis
from ..inpainting.patchmatch import PatchMatchInpainting
from ..inpainting.basic import BasicInpainting


class PhotoRefineGUI:
    """Complete GUI with all features properly exposed"""

    def __init__(self, root):
        self.root = root
        self.root.title("PhotoRefine Pro - Advanced Glare & Reflection Removal")
        self.root.geometry("1700x1000")
        self.root.minsize(1500, 900)

        self.processor = BlobRemovalProcessor()
        self.current_image_path = None
        self.processing = False
        self.histogram_window = None

        # Complete parameters
        self.params = {
            # Detection method
            'detection_method': tk.StringVar(value='brightness'),

            # HSV parameters
            'h_min': tk.IntVar(value=0),
            'h_max': tk.IntVar(value=179),
            's_min': tk.IntVar(value=0),
            's_max': tk.IntVar(value=255),
            'v_min': tk.IntVar(value=200),
            'v_max': tk.IntVar(value=255),

            # Basic thresholds
            'brightness_threshold': tk.IntVar(value=220),
            'saturation_threshold': tk.IntVar(value=100),

            # Edge detection
            'canny_low': tk.IntVar(value=50),
            'canny_high': tk.IntVar(value=150),

            # Adaptive threshold
            'block_size': tk.IntVar(value=11),
            'c_value': tk.IntVar(value=2),

            # K-means
            'n_clusters': tk.IntVar(value=5),

            # Watershed parameters
            'watershed_markers': tk.IntVar(value=5),
            'watershed_compactness': tk.DoubleVar(value=0.001),

            # DFT parameters
            'dft_filter_type': tk.StringVar(value='notch'),
            'dft_cutoff': tk.IntVar(value=30),
            'dft_radius': tk.IntVar(value=10),
            'dft_auto_detect': tk.BooleanVar(value=True),

            # Contrast parameters
            'contrast_method': tk.StringVar(value='none'),
            'gamma': tk.DoubleVar(value=1.0),
            'clahe_clip': tk.DoubleVar(value=2.0),
            'clahe_tile_size': tk.IntVar(value=8),

            # Morphology
            'morph_operation': tk.StringVar(value='close'),
            'kernel_size': tk.IntVar(value=5),
            'morph_iterations': tk.IntVar(value=1),

            # Area filtering
            'min_area': tk.IntVar(value=100),
            'max_area': tk.IntVar(value=50000),

            # Inpainting
            'inpaint_method': tk.StringVar(value='telea'),
            'inpaint_radius': tk.IntVar(value=5),
            'enable_inpaint': tk.BooleanVar(value=True),

            # PatchMatch
            'patchmatch_patch_size': tk.IntVar(value=7),
            'patchmatch_iterations': tk.IntVar(value=5),
            'patchmatch_alpha': tk.DoubleVar(value=0.5),

            # Processing mode
            'use_working_image': tk.BooleanVar(value=True),
        }

        self.setup_ui()
        self.update_history_ui()

    def setup_ui(self):
        """Setup the complete user interface"""
        # Main container with notebook for tabs
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))

        # Tab 1: Detection & Filtering
        detection_frame = ttk.Frame(self.notebook, width=400)
        self.notebook.add(detection_frame, text="Detection & Filtering")
        self._build_detection_tab(detection_frame)

        # Tab 2: Inpainting
        inpainting_frame = ttk.Frame(self.notebook, width=400)
        self.notebook.add(inpainting_frame, text="Inpainting")
        self._build_inpainting_tab(inpainting_frame)

        # Tab 3: Contrast & Histogram
        contrast_frame = ttk.Frame(self.notebook, width=400)
        self.notebook.add(contrast_frame, text="Contrast & Histogram")
        self._build_contrast_tab(contrast_frame)

        # RIGHT PANEL - Image Display
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # History controls at top
        history_frame = ttk.LabelFrame(right_frame, text="Processing History", padding="5")
        history_frame.pack(fill=tk.X, padx=5, pady=5)
        self._build_history_controls(history_frame)

        # Image display area
        images_frame = ttk.LabelFrame(right_frame, text="Results", padding="5")
        images_frame.pack(fill=tk.BOTH, expand=True)

        # Create grid for images
        grid_frame = ttk.Frame(images_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(grid_frame, text="Original", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Label(grid_frame, text="Working Image", font=('Arial', 11, 'bold')).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Label(grid_frame, text="Mask Preview", font=('Arial', 11, 'bold')).grid(
            row=0, column=2, padx=5, pady=5)

        self.original_canvas = tk.Canvas(grid_frame, bg='#2b2b2b', width=400, height=400)
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5, sticky=tk.NSEW)

        self.working_canvas = tk.Canvas(grid_frame, bg='#2b2b2b', width=400, height=400)
        self.working_canvas.grid(row=1, column=1, padx=5, pady=5, sticky=tk.NSEW)

        self.mask_canvas = tk.Canvas(grid_frame, bg='#2b2b2b', width=400, height=400)
        self.mask_canvas.grid(row=1, column=2, padx=5, pady=5, sticky=tk.NSEW)

        for i in range(3):
            grid_frame.columnconfigure(i, weight=1)
        grid_frame.rowconfigure(1, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=3)

    def _build_detection_tab(self, parent):
        """Build detection and filtering tab"""
        # Scrollable canvas
        canvas = tk.Canvas(parent, highlightthickness=0, width=380)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0

        # File operations
        ttk.Label(scroll_frame, text="File Operations", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        ttk.Button(scroll_frame, text="📁 Load Image",
                  command=self.load_image).grid(row=row, column=0, columnspan=2,
                                               sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Button(scroll_frame, text="💾 Save Result",
                  command=self.save_image).grid(row=row, column=0, columnspan=2,
                                               sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Processing Mode
        ttk.Label(scroll_frame, text="Processing Mode", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        ttk.Radiobutton(scroll_frame, text="Process from Original",
                       variable=self.params['use_working_image'],
                       value=False).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
        row += 1

        ttk.Radiobutton(scroll_frame, text="Process from Current (Multi-pass)",
                       variable=self.params['use_working_image'],
                       value=True).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
        row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Detection Method
        ttk.Label(scroll_frame, text="Detection Method", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        methods = [
            ('Brightness Threshold', 'brightness'),
            ('HSV Color Range', 'hsv'),
            ('Saturation Threshold', 'saturation'),
            ('Edge Detection (Canny)', 'edge'),
            ('Adaptive Threshold', 'adaptive'),
            ('K-Means Clustering', 'kmeans'),
            ('🆕 Watershed Segmentation', 'watershed'),
            ('🆕 DFT Frequency Filter', 'dft'),
        ]

        for label, value in methods:
            ttk.Radiobutton(scroll_frame, text=label, variable=self.params['detection_method'],
                           value=value, command=self.on_method_change).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
            row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Method Parameters (dynamic frame)
        ttk.Label(scroll_frame, text="Method Parameters", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        self.method_params_frame = ttk.Frame(scroll_frame)
        self.method_params_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Morphological Operations
        ttk.Label(scroll_frame, text="Morphological Operations", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        morph_options = [('None', 'none'), ('Open', 'open'), ('Close', 'close'), ('Dilate', 'dilate'), ('Erode', 'erode')]
        for i, (label, value) in enumerate(morph_options):
            if i % 2 == 0:
                col = 0
            else:
                col = 1
            ttk.Radiobutton(scroll_frame, text=label, variable=self.params['morph_operation'],
                           value=value, command=self.update_mask).grid(
                row=row + i//2, column=col, sticky=tk.W, padx=20, pady=2)

        row += (len(morph_options) + 1) // 2

        row = self._add_slider(scroll_frame, row, "Kernel Size", self.params['kernel_size'], 1, 51, step=2)
        row = self._add_slider(scroll_frame, row, "Iterations", self.params['morph_iterations'], 1, 10)

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Area Filtering
        ttk.Label(scroll_frame, text="Area Filtering", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        row = self._add_slider(scroll_frame, row, "Min Area (px)", self.params['min_area'], 0, 10000)
        row = self._add_slider(scroll_frame, row, "Max Area (px)", self.params['max_area'], 1000, 500000)

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Action Buttons
        ttk.Button(scroll_frame, text="🔍 Preview Mask Only",
                  command=self.update_mask).grid(row=row, column=0, columnspan=2,
                                                 sticky=tk.EW, padx=5, pady=5)
        row += 1

        ttk.Button(scroll_frame, text="✨ Apply Filter & Inpaint",
                  command=self.process_image,
                  style='Accent.TButton').grid(row=row, column=0, columnspan=2,
                                                   sticky=tk.EW, padx=5, pady=5)

        # Populate initial method params
        self.populate_method_params()

    def _build_inpainting_tab(self, parent):
        """Build inpainting configuration tab"""
        # Scrollable canvas
        canvas = tk.Canvas(parent, highlightthickness=0, width=380)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0

        # Enable/Disable Inpainting
        ttk.Label(scroll_frame, text="Inpainting Settings", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        row += 1

        ttk.Checkbutton(scroll_frame, text="Enable Inpainting",
                       variable=self.params['enable_inpaint'],
                       command=self.on_inpaint_toggle).grid(row=row, column=0, columnspan=2,
                                                      sticky=tk.W, padx=20, pady=5)
        row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Inpainting Method Selection
        ttk.Label(scroll_frame, text="Inpainting Method", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        inpaint_methods = [
            ('Telea (Fast Marching)', 'telea'),
            ('Navier-Stokes', 'ns'),
            ('Bilateral Filter', 'bilateral'),
            ('Morphological', 'morphological'),
            ('Multiscale Decomposition', 'multiscale'),
            ('🆕 PatchMatch (Best Quality)', 'patchmatch'),
        ]

        for label, value in inpaint_methods:
            ttk.Radiobutton(scroll_frame, text=label, variable=self.params['inpaint_method'],
                           value=value, command=self.on_inpaint_method_change).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
            row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Basic Inpainting Parameters
        ttk.Label(scroll_frame, text="Basic Inpainting Parameters", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        row = self._add_slider(scroll_frame, row, "Inpaint Radius", self.params['inpaint_radius'], 1, 50)

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # PatchMatch Parameters
        ttk.Label(scroll_frame, text="PatchMatch Parameters", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        ttk.Label(scroll_frame, text="(Only used when PatchMatch is selected)",
                 font=('Arial', 8, 'italic'), foreground='gray').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
        row += 1

        row = self._add_slider(scroll_frame, row, "Patch Size", self.params['patchmatch_patch_size'], 3, 15, step=2)
        row = self._add_slider(scroll_frame, row, "Iterations", self.params['patchmatch_iterations'], 1, 10)
        row = self._add_slider(scroll_frame, row, "Guidance (α)", self.params['patchmatch_alpha'], 0, 1, step=0.1)

        ttk.Label(scroll_frame, text="💡 Tip: Higher guidance follows edges more closely",
                 font=('Arial', 8), foreground='blue').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=5)
        row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Info text
        info_text = tk.Text(scroll_frame, height=10, width=45, wrap=tk.WORD, font=('Arial', 9))
        info_text.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        info_text.insert('1.0', """Inpainting Methods:

• Telea: Fast, good for small regions
• Navier-Stokes: Better for large areas
• Bilateral: Edge-preserving
• Morphological: Structure-preserving
• Multiscale: Base+detail separation
• PatchMatch: Best quality, slower
  (with structure guidance)""")
        info_text.config(state=tk.DISABLED)

    def _build_contrast_tab(self, parent):
        """Build contrast and histogram analysis tab"""
        # Scrollable canvas
        canvas = tk.Canvas(parent, highlightthickness=0, width=380)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0

        # Histogram Analysis
        ttk.Label(scroll_frame, text="Histogram Analysis", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        row += 1

        ttk.Button(scroll_frame, text="📊 Show Histogram",
                  command=self.show_histogram).grid(row=row, column=0, columnspan=2,
                                                    sticky=tk.EW, padx=5, pady=5)
        row += 1

        ttk.Button(scroll_frame, text="🔍 Analyze Image Statistics",
                  command=self.analyze_statistics).grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, padx=5, pady=5)
        row += 1

        ttk.Button(scroll_frame, text="💡 Detect Glare (Histogram)",
                  command=self.detect_glare_histogram).grid(row=row, column=0, columnspan=2,
                                                            sticky=tk.EW, padx=5, pady=5)
        row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Contrast Enhancement
        ttk.Label(scroll_frame, text="Contrast Enhancement", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        row += 1

        contrast_methods = [
            ('None', 'none'),
            ('Histogram Equalization', 'hist_eq'),
            ('CLAHE (Adaptive)', 'clahe'),
            ('Gamma Correction', 'gamma'),
            ('Linear Stretch', 'linear'),
            ('Sigmoid Contrast', 'sigmoid'),
            ('Local Enhancement', 'local'),
            ('Auto Contrast', 'auto'),
        ]

        for label, value in contrast_methods:
            ttk.Radiobutton(scroll_frame, text=label, variable=self.params['contrast_method'],
                           value=value, command=self.on_contrast_change).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
            row += 1

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Contrast Parameters
        ttk.Label(scroll_frame, text="Contrast Parameters", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        row = self._add_slider(scroll_frame, row, "Gamma", self.params['gamma'], 0.1, 3.0, step=0.1)
        row = self._add_slider(scroll_frame, row, "CLAHE Clip Limit", self.params['clahe_clip'], 1.0, 10.0, step=0.5)
        row = self._add_slider(scroll_frame, row, "CLAHE Tile Size", self.params['clahe_tile_size'], 2, 16)

        ttk.Separator(scroll_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=10, padx=5)
        row += 1

        # Apply Contrast Button
        ttk.Button(scroll_frame, text="✨ Apply Contrast Enhancement",
                  command=self.apply_contrast,
                  style='Accent.TButton').grid(row=row, column=0, columnspan=2,
                                                   sticky=tk.EW, padx=5, pady=10)
        row += 1

        # Info
        info_text = tk.Text(scroll_frame, height=8, width=45, wrap=tk.WORD, font=('Arial', 9))
        info_text.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        info_text.insert('1.0', """Contrast Methods:

• Histogram Eq: Global contrast
• CLAHE: Local adaptive contrast
• Gamma: Brightness/darkness
• Linear: Stretch to full range
• Auto: Automatic adjustment

Use before detection for better results!""")
        info_text.config(state=tk.DISABLED)

    def _build_history_controls(self, parent):
        """Build history management controls"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)

        self.undo_btn = ttk.Button(button_frame, text="⬅ Undo",
                                    command=self.undo_step, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=2)

        self.redo_btn = ttk.Button(button_frame, text="➡ Redo",
                                    command=self.redo_step, state=tk.DISABLED)
        self.redo_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(button_frame, text="↺ Reset",
                  command=self.reset_to_original).pack(side=tk.LEFT, padx=2)

        self.history_label = ttk.Label(button_frame, text="No history",
                                       font=('Arial', 9))
        self.history_label.pack(side=tk.RIGHT, padx=5)

    def populate_method_params(self):
        """Populate method-specific parameters dynamically"""
        # Clear existing widgets
        for widget in self.method_params_frame.winfo_children():
            widget.destroy()

        method = self.params['detection_method'].get()
        row = 0

        if method == 'hsv':
            row = self._add_slider(self.method_params_frame, row, "Hue Min", self.params['h_min'], 0, 179)
            row = self._add_slider(self.method_params_frame, row, "Hue Max", self.params['h_max'], 0, 179)
            row = self._add_slider(self.method_params_frame, row, "Sat Min", self.params['s_min'], 0, 255)
            row = self._add_slider(self.method_params_frame, row, "Sat Max", self.params['s_max'], 0, 255)
            row = self._add_slider(self.method_params_frame, row, "Val Min", self.params['v_min'], 0, 255)
            row = self._add_slider(self.method_params_frame, row, "Val Max", self.params['v_max'], 0, 255)

        elif method == 'brightness':
            row = self._add_slider(self.method_params_frame, row, "Threshold",
                                  self.params['brightness_threshold'], 0, 255)
            ttk.Label(self.method_params_frame, text="💡 Higher = only very bright spots",
                     font=('Arial', 8), foreground='gray').grid(
                row=row, column=0, columnspan=2, pady=2)

        elif method == 'saturation':
            row = self._add_slider(self.method_params_frame, row, "Threshold",
                                  self.params['saturation_threshold'], 0, 255)

        elif method == 'edge':
            row = self._add_slider(self.method_params_frame, row, "Low Threshold",
                                  self.params['canny_low'], 0, 500)
            row = self._add_slider(self.method_params_frame, row, "High Threshold",
                                  self.params['canny_high'], 0, 500)

        elif method == 'adaptive':
            row = self._add_slider(self.method_params_frame, row, "Block Size",
                                  self.params['block_size'], 3, 99, step=2)
            row = self._add_slider(self.method_params_frame, row, "C Value",
                                  self.params['c_value'], -20, 20)

        elif method == 'kmeans':
            row = self._add_slider(self.method_params_frame, row, "Clusters",
                                  self.params['n_clusters'], 2, 20)

        elif method == 'watershed':
            ttk.Label(self.method_params_frame, text="🆕 Watershed Segmentation",
                     font=('Arial', 10, 'bold'), foreground='blue').grid(
                row=row, column=0, columnspan=2, pady=5)
            row += 1
            row = self._add_slider(self.method_params_frame, row, "Markers",
                                  self.params['watershed_markers'], 2, 20)
            row = self._add_slider(self.method_params_frame, row, "Compactness",
                                  self.params['watershed_compactness'], 0.0001, 0.01, step=0.0001)
            ttk.Label(self.method_params_frame, text="Best for color-based segmentation",
                     font=('Arial', 8), foreground='gray').grid(
                row=row, column=0, columnspan=2, pady=2)

        elif method == 'dft':
            ttk.Label(self.method_params_frame, text="🆕 DFT Frequency Filtering",
                     font=('Arial', 10, 'bold'), foreground='blue').grid(
                row=row, column=0, columnspan=2, pady=5)
            row += 1

            ttk.Label(self.method_params_frame, text="Filter Type:").grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=2)
            dft_combo = ttk.Combobox(self.method_params_frame, textvariable=self.params['dft_filter_type'],
                                    values=['notch', 'highpass', 'bandreject', 'adaptive'],
                                    state='readonly', width=15)
            dft_combo.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            row += 1

            ttk.Checkbutton(self.method_params_frame, text="Auto-detect frequencies",
                           variable=self.params['dft_auto_detect']).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
            row += 1

            row = self._add_slider(self.method_params_frame, row, "Cutoff Freq",
                                  self.params['dft_cutoff'], 10, 100)
            row = self._add_slider(self.method_params_frame, row, "Radius",
                                  self.params['dft_radius'], 5, 50)
            ttk.Label(self.method_params_frame, text="Best for uniform/periodic reflections",
                     font=('Arial', 8), foreground='gray').grid(
                row=row, column=0, columnspan=2, pady=2)

    def _add_slider(self, parent, row, label, variable, from_, to, step=1):
        """Add a labeled slider"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)

        label_w = ttk.Label(frame, text=label, width=15)
        label_w.pack(side=tk.LEFT)

        value_label = ttk.Label(frame, text=str(variable.get()), width=7)
        value_label.pack(side=tk.RIGHT)

        def update_label(val):
            if step < 1:
                value_label.config(text=f"{float(val):.3f}")
            else:
                value_label.config(text=str(int(float(val))))
            if self.processor.original_image is not None and not self.processing:
                self.root.after(100, self.update_mask)

        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                          variable=variable, command=update_label)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        return row + 1

    def on_method_change(self):
        """Handle detection method change"""
        self.populate_method_params()
        self.update_mask()

    def on_inpaint_toggle(self):
        """Handle inpainting enable/disable"""
        pass

    def on_inpaint_method_change(self):
        """Handle inpainting method change"""
        pass

    def on_contrast_change(self):
        """Handle contrast method change"""
        pass

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.current_image_path = file_path
                image = self.processor.load_image(file_path)
                self.display_image(image, self.original_canvas)
                self.display_image(image, self.working_canvas)
                self.status_var.set(f"✓ Loaded: {os.path.basename(file_path)}")
                self.update_history_ui()
                self.update_mask()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set(f"✗ Error loading image")

    def save_image(self):
        """Save the processed image"""
        if self.processor.processed_image is None:
            messagebox.showwarning("No Image", "No processed image to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )

        if file_path:
            try:
                image_bgr = cv2.cvtColor(self.processor.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, image_bgr)
                self.status_var.set(f"✓ Saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def reset_to_original(self):
        """Reset to original image"""
        if self.processor.reset_to_original():
            image = self.processor.get_working_image()
            self.display_image(image, self.working_canvas)
            self.mask_canvas.delete("all")
            self.status_var.set("✓ Reset to original image")
            self.update_history_ui()

    def undo_step(self):
        """Undo last processing step"""
        success, step, total = self.processor.undo_step()
        if success:
            image = self.processor.get_working_image()
            self.display_image(image, self.working_canvas)
            if self.processor.current_mask is not None:
                mask_colored = cv2.cvtColor(self.processor.current_mask, cv2.COLOR_GRAY2RGB)
                mask_colored[self.processor.current_mask > 0] = [255, 200, 0]
                self.display_image(mask_colored, self.mask_canvas)
            self.status_var.set(f"✓ Undo successful")
            self.update_history_ui()

    def redo_step(self):
        """Redo next processing step"""
        success, step, total = self.processor.redo_step()
        if success:
            image = self.processor.get_working_image()
            self.display_image(image, self.working_canvas)
            if self.processor.current_mask is not None:
                mask_colored = cv2.cvtColor(self.processor.current_mask, cv2.COLOR_GRAY2RGB)
                mask_colored[self.processor.current_mask > 0] = [255, 200, 0]
                self.display_image(mask_colored, self.mask_canvas)
            self.status_var.set(f"✓ Redo successful")
            self.update_history_ui()

    def update_history_ui(self):
        """Update history UI elements"""
        info = self.processor.get_history_info()

        if info['total_steps'] > 0:
            self.history_label.config(
                text=f"Step {info['current_step'] + 1}/{info['total_steps']}"
            )
        else:
            self.history_label.config(text="No history")

        self.undo_btn.config(state=tk.NORMAL if info['can_undo'] else tk.DISABLED)
        self.redo_btn.config(state=tk.NORMAL if info['can_redo'] else tk.DISABLED)

    def update_mask(self):
        """Update the mask preview (without applying)"""
        if self.processor.original_image is None or self.processing:
            return

        self.processing = True
        self.status_var.set("⏳ Generating mask preview...")
        self.root.update()

        def process():
            try:
                params = self.get_current_params()
                params_copy = params.copy()
                params_copy['enable_inpaint'] = False
                use_working = params['use_working_image']

                # Special handling for DFT method
                if params['detection_method'] == 'dft':
                    image = self.processor.working_image if use_working else self.processor.original_image
                    result = self.apply_dft_filter(image, params)
                    # Create a difference mask
                    diff = cv2.absdiff(image, result)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                    _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
                else:
                    _, mask = self.processor.process_with_params(
                        params_copy,
                        use_working_image=use_working,
                        add_to_history=False
                    )

                self.root.after(0, lambda: self._display_mask(mask))
            except Exception as e:
                self.root.after(0, lambda: self._handle_error(str(e)))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _display_mask(self, mask):
        """Display mask preview"""
        if mask is not None:
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_colored[mask > 0] = [255, 200, 0]
            self.display_image(mask_colored, self.mask_canvas)
        self.status_var.set("✓ Ready")
        self.processing = False

    def _handle_error(self, error_msg):
        """Handle processing errors"""
        self.status_var.set(f"✗ Error: {error_msg}")
        self.processing = False
        messagebox.showerror("Processing Error", error_msg)

    def apply_dft_filter(self, image, params):
        """Apply DFT filter to image"""
        filter_type = params.get('dft_filter_type', 'notch')
        cutoff = params.get('dft_cutoff', 30)
        radius = params.get('dft_radius', 10)
        auto_detect = params.get('dft_auto_detect', True)

        if filter_type == 'notch':
            return DFTFilter.notch_filter(image, radius=radius, auto_detect=auto_detect)
        elif filter_type == 'highpass':
            return DFTFilter.highpass_filter(image, cutoff_freq=cutoff)
        elif filter_type == 'bandreject':
            return DFTFilter.bandreject_filter(image, center_freq=cutoff, bandwidth=radius)
        elif filter_type == 'adaptive':
            return DFTFilter.adaptive_frequency_filter(image, radius=radius)
        else:
            return image

    def process_image(self):
        """Process the image and apply inpainting"""
        if self.processor.original_image is None or self.processing:
            return

        self.processing = True
        mode = "current result" if self.params['use_working_image'].get() else "original"
        self.status_var.set(f"⏳ Applying filter to {mode}...")
        self.root.update()

        def process():
            try:
                params = self.get_current_params()
                use_working = params['use_working_image']

                # Special handling for DFT and PatchMatch
                if params['detection_method'] == 'dft':
                    image = self.processor.working_image if use_working else self.processor.original_image
                    result = self.apply_dft_filter(image, params)
                    diff = cv2.absdiff(image, result)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                    _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)

                    # Apply inpainting if enabled
                    if params.get('enable_inpaint', False):
                        if params.get('inpaint_method') == 'patchmatch':
                            result = PatchMatchInpainting.inpaint(
                                result, mask,
                                patch_size=params.get('patchmatch_patch_size', 7),
                                iterations=params.get('patchmatch_iterations', 5),
                                alpha=params.get('patchmatch_alpha', 0.5)
                            )
                        else:
                            result = BasicInpainting.inpaint(
                                result, mask,
                                method=params.get('inpaint_method', 'telea'),
                                radius=params.get('inpaint_radius', 5)
                            )

                    self.processor.processed_image = result
                    self.processor.working_image = result.copy()
                    self.processor.current_mask = mask
                    self.processor.add_to_history(params, mask, result)

                elif params.get('inpaint_method') == 'patchmatch' and params.get('enable_inpaint', False):
                    # Get mask first
                    _, mask = self.processor.process_with_params(
                        {**params, 'enable_inpaint': False},
                        use_working_image=use_working,
                        add_to_history=False
                    )

                    # Apply PatchMatch inpainting
                    image = self.processor.working_image if use_working else self.processor.original_image
                    result = PatchMatchInpainting.inpaint(
                        image, mask,
                        patch_size=params.get('patchmatch_patch_size', 7),
                        iterations=params.get('patchmatch_iterations', 5),
                        alpha=params.get('patchmatch_alpha', 0.5)
                    )

                    self.processor.processed_image = result
                    self.processor.working_image = result.copy()
                    self.processor.current_mask = mask
                    self.processor.add_to_history(params, mask, result)
                else:
                    result, mask = self.processor.process_with_params(
                        params,
                        use_working_image=use_working,
                        add_to_history=True
                    )

                self.root.after(0, lambda: self._display_result(result, mask))
            except Exception as e:
                self.root.after(0, lambda: self._handle_error(str(e)))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _display_result(self, result, mask):
        """Display result"""
        if result is not None:
            self.display_image(result, self.working_canvas)
            if mask is not None:
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                mask_colored[mask > 0] = [255, 200, 0]
                self.display_image(mask_colored, self.mask_canvas)
            self.status_var.set("✓ Filter applied successfully!")
            self.update_history_ui()
        else:
            self.status_var.set("✗ Processing failed!")
        self.processing = False

    def apply_contrast(self):
        """Apply contrast enhancement"""
        if self.processor.working_image is None:
            messagebox.showwarning("No Image", "Load an image first!")
            return

        method = self.params['contrast_method'].get()
        if method == 'none':
            return

        try:
            image = self.processor.working_image.copy()

            if method == 'hist_eq':
                result = ContrastEnhancement.histogram_equalization(image)
            elif method == 'clahe':
                clip = self.params['clahe_clip'].get()
                tile_size = self.params['clahe_tile_size'].get()
                result = ContrastEnhancement.adaptive_histogram_equalization(
                    image, clip_limit=clip, tile_size=tile_size)
            elif method == 'gamma':
                gamma = self.params['gamma'].get()
                result = ContrastEnhancement.gamma_correction(image, gamma=gamma)
            elif method == 'linear':
                result = ContrastEnhancement.linear_contrast_stretch(image)
            elif method == 'sigmoid':
                result = ContrastEnhancement.sigmoid_contrast(image)
            elif method == 'local':
                result = ContrastEnhancement.local_contrast_enhancement(image)
            elif method == 'auto':
                result = ContrastEnhancement.auto_contrast(image)
            else:
                return

            self.processor.working_image = result
            self.processor.processed_image = result
            self.display_image(result, self.working_canvas)
            self.status_var.set(f"✓ Applied {method} contrast enhancement")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply contrast: {str(e)}")

    def show_histogram(self):
        """Show histogram visualization"""
        if self.processor.working_image is None:
            messagebox.showwarning("No Image", "Load an image first!")
            return

        try:
            hist_img = HistogramAnalysis.create_histogram_image(
                self.processor.working_image, size=(512, 400))

            # Create new window
            hist_window = tk.Toplevel(self.root)
            hist_window.title("Image Histogram")

            pil_img = Image.fromarray(hist_img)
            photo = ImageTk.PhotoImage(pil_img)

            label = tk.Label(hist_window, image=photo)
            label.image = photo
            label.pack(padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show histogram: {str(e)}")

    def analyze_statistics(self):
        """Analyze and display image statistics"""
        if self.processor.working_image is None:
            messagebox.showwarning("No Image", "Load an image first!")
            return

        try:
            stats = HistogramAnalysis.analyze_histogram(self.processor.working_image)
            suggestions = HistogramAnalysis.suggest_contrast_adjustment(self.processor.working_image)

            # Create info window
            info_window = tk.Toplevel(self.root)
            info_window.title("Image Statistics")
            info_window.geometry("500x400")

            text = scrolledtext.ScrolledText(info_window, width=60, height=20, font=('Courier', 10))
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            output = "IMAGE STATISTICS\n" + "="*50 + "\n\n"

            for channel_name, channel_stats in stats.items():
                output += f"{channel_name} Channel:\n"
                output += f"  Mean: {channel_stats['mean']:.2f}\n"
                output += f"  Std Dev: {channel_stats['std']:.2f}\n"
                output += f"  Min: {channel_stats['min']}\n"
                output += f"  Max: {channel_stats['max']}\n"
                output += f"  Median: {channel_stats['median']:.2f}\n"
                output += f"  Mode: {channel_stats['mode']}\n"
                output += f"  Dynamic Range: {channel_stats['dynamic_range']}\n\n"

            output += "\nCONTRAST ANALYSIS\n" + "="*50 + "\n\n"
            output += f"Dynamic Range: {suggestions['dynamic_range']}\n"
            output += f"Standard Deviation: {suggestions['std_dev']:.2f}\n"
            output += f"Mean Brightness: {suggestions['mean']:.2f}\n\n"

            output += "SUGGESTIONS:\n"
            for suggestion in suggestions['suggestions']:
                output += f"  • {suggestion}\n"

            text.insert('1.0', output)
            text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze statistics: {str(e)}")

    def detect_glare_histogram(self):
        """Detect glare using histogram analysis"""
        if self.processor.working_image is None:
            messagebox.showwarning("No Image", "Load an image first!")
            return

        try:
            result = HistogramAnalysis.detect_glare_from_histogram(self.processor.working_image)

            message = f"GLARE DETECTION RESULTS\n\n"
            message += f"Has Glare: {'YES' if result['has_glare'] else 'NO'}\n\n"
            message += f"Bright Pixels: {result['bright_pixel_count']}\n"
            message += f"Bright Pixel Ratio: {result['bright_pixel_ratio']:.2%}\n"
            message += f"Threshold Used: {result['brightness_threshold']}\n\n"
            message += f"Recommended Threshold: {result['recommended_threshold']:.0f}\n"
            message += f"Mean Brightness: {result['mean_brightness']:.2f}\n"
            message += f"Std Brightness: {result['std_brightness']:.2f}\n"

            if result['has_glare']:
                message += f"\n💡 Suggestion: Use brightness detection with threshold {result['recommended_threshold']:.0f}"

            messagebox.showinfo("Glare Detection", message)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect glare: {str(e)}")

    def get_current_params(self):
        """Get current parameter values"""
        return {k: v.get() for k, v in self.params.items()}

    def display_image(self, image, canvas):
        """Display an image on a canvas"""
        if image is None:
            return

        height, width = image.shape[:2]
        max_size = 400

        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(pil_image)

        canvas.delete("all")
        canvas.create_image(canvas.winfo_width()//2, canvas.winfo_height()//2, image=photo)
        canvas.image = photo


def main():
    """Main entry point"""
    root = tk.Tk()

    # Configure accent button style
    style = ttk.Style()
    try:
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    except:
        pass

    app = PhotoRefineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
