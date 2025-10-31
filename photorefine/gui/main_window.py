"""
PhotoRefine: Enhanced GUI Module with advanced features
User interface for interactive glare/reflection removal with:
- Multiple detection methods including Watershed and DFT
- PatchMatch inpainting support
- Multi-pass filtering support
"""

import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import threading
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.blob_processor import BlobRemovalProcessor
from ..filters.color_segmentation import WatershedSegmentation
from ..filters.fourier import DFTFilter
from ..inpainting.patchmatch import PatchMatchInpainting


class PhotoRefineGUI:
    """Enhanced GUI with watershed, DFT, and PatchMatch support"""

    def __init__(self, root):
        self.root = root
        self.root.title("PhotoRefine - Advanced Glare & Reflection Removal")
        self.root.geometry("1600x950")
        self.root.minsize(1400, 800)

        self.processor = BlobRemovalProcessor()
        self.current_image_path = None
        self.processing = False

        # Enhanced parameters
        self.params = {
            'detection_method': tk.StringVar(value='hsv'),
            # HSV parameters
            'h_min': tk.IntVar(value=15),
            'h_max': tk.IntVar(value=45),
            's_min': tk.IntVar(value=80),
            's_max': tk.IntVar(value=255),
            'v_min': tk.IntVar(value=100),
            'v_max': tk.IntVar(value=255),
            # Other detection parameters
            'brightness_threshold': tk.IntVar(value=200),
            'saturation_threshold': tk.IntVar(value=100),
            'canny_low': tk.IntVar(value=50),
            'canny_high': tk.IntVar(value=150),
            'block_size': tk.IntVar(value=11),
            'c_value': tk.IntVar(value=2),
            'n_clusters': tk.IntVar(value=5),
            # Watershed parameters
            'watershed_markers': tk.IntVar(value=5),
            'watershed_compactness': tk.DoubleVar(value=0.001),
            # DFT parameters
            'dft_filter_type': tk.StringVar(value='notch'),
            'dft_cutoff': tk.IntVar(value=30),
            'dft_radius': tk.IntVar(value=10),
            # Morphology parameters
            'morph_operation': tk.StringVar(value='close'),
            'kernel_size': tk.IntVar(value=5),
            'morph_iterations': tk.IntVar(value=1),
            # Area filtering
            'min_area': tk.IntVar(value=100),
            'max_area': tk.IntVar(value=50000),
            # Inpainting parameters
            'inpaint_method': tk.StringVar(value='telea'),
            'inpaint_radius': tk.IntVar(value=5),
            'enable_inpaint': tk.BooleanVar(value=True),
            'use_working_image': tk.BooleanVar(value=True),
            # PatchMatch parameters
            'patchmatch_patch_size': tk.IntVar(value=7),
            'patchmatch_iterations': tk.IntVar(value=5),
            'patchmatch_alpha': tk.DoubleVar(value=0.5),
        }

        self.setup_ui()
        self.update_history_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # LEFT PANEL - Controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))

        # Scrollable control panel
        canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        control_frame = ttk.Frame(canvas)

        control_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=control_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set, width=350)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Control panel contents
        self._build_controls(control_frame)

        # RIGHT PANEL - Image Display
        right_frame = ttk.Frame(main_frame)
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

        ttk.Label(grid_frame, text="Original", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Label(grid_frame, text="Working Image", font=('Arial', 10, 'bold')).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Label(grid_frame, text="Mask Preview", font=('Arial', 10, 'bold')).grid(
            row=0, column=2, padx=5, pady=5)

        self.original_canvas = tk.Canvas(grid_frame, bg='#2b2b2b')
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5, sticky=tk.NSEW)

        self.working_canvas = tk.Canvas(grid_frame, bg='#2b2b2b')
        self.working_canvas.grid(row=1, column=1, padx=5, pady=5, sticky=tk.NSEW)

        self.mask_canvas = tk.Canvas(grid_frame, bg='#2b2b2b')
        self.mask_canvas.grid(row=1, column=2, padx=5, pady=5, sticky=tk.NSEW)

        for i in range(3):
            grid_frame.columnconfigure(i, weight=1)
        grid_frame.rowconfigure(1, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=3)

    def _build_history_controls(self, parent):
        """Build history management controls"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)

        self.undo_btn = ttk.Button(button_frame, text="← Undo",
                                    command=self.undo_step, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=2)

        self.redo_btn = ttk.Button(button_frame, text="→ Redo",
                                    command=self.redo_step, state=tk.DISABLED)
        self.redo_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(button_frame, text="↺ Reset to Original",
                  command=self.reset_to_original).pack(side=tk.LEFT, padx=2)

        self.history_label = ttk.Label(button_frame, text="No history",
                                       font=('Arial', 9))
        self.history_label.pack(side=tk.RIGHT, padx=5)

    def _build_controls(self, parent):
        """Build enhanced control panel"""
        row = 0

        # Title
        ttk.Label(parent, text="PhotoRefine Pro", font=('Arial', 14, 'bold')).grid(
            row=row, column=0, columnspan=2, pady=10)
        row += 1

        # File operations
        ttk.Button(parent, text="📁 Load Image",
                  command=self.load_image).grid(row=row, column=0, columnspan=2,
                                               sticky=tk.EW, padx=5, pady=3)
        row += 1

        ttk.Button(parent, text="💾 Save Result",
                  command=self.save_image).grid(row=row, column=0, columnspan=2,
                                               sticky=tk.EW, padx=5, pady=3)
        row += 1

        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1

        # Processing Mode
        ttk.Label(parent, text="Processing Mode", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        mode_frame = ttk.Frame(parent)
        mode_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)

        ttk.Radiobutton(mode_frame, text="From Original",
                       variable=self.params['use_working_image'],
                       value=False).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(mode_frame, text="From Current",
                       variable=self.params['use_working_image'],
                       value=True).pack(side=tk.LEFT, padx=3)
        row += 1

        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1

        # Detection method
        ttk.Label(parent, text="Detection Method", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        methods = [
            ('HSV Color', 'hsv'),
            ('Brightness', 'brightness'),
            ('Saturation', 'saturation'),
            ('Edge Detection', 'edge'),
            ('Adaptive', 'adaptive'),
            ('K-Means', 'kmeans'),
            ('Watershed (NEW)', 'watershed'),
            ('DFT Filter (NEW)', 'dft')
        ]

        for label, value in methods:
            ttk.Radiobutton(parent, text=label, variable=self.params['detection_method'],
                           value=value, command=self.on_method_change).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
            row += 1

        # Method-specific parameters frame (will be populated dynamically)
        self.method_params_frame = ttk.LabelFrame(parent, text="Method Parameters", padding="5")
        self.method_params_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        row += 1

        # Initially populate with HSV parameters
        self.current_method_row = 0
        self.populate_method_params()

        # Morphology section
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1

        ttk.Label(parent, text="Morphology", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        morph_frame = ttk.Frame(parent)
        morph_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)

        for label, value in [('None', 'none'), ('Open', 'open'),
                             ('Close', 'close'), ('Dilate', 'dilate')]:
            ttk.Radiobutton(morph_frame, text=label, variable=self.params['morph_operation'],
                           value=value, command=self.update_mask).pack(side=tk.LEFT, padx=3)
        row += 1

        row = self._add_slider(parent, row, "Kernel Size", self.params['kernel_size'], 1, 51, step=2)
        row = self._add_slider(parent, row, "Iterations", self.params['morph_iterations'], 1, 10)

        # Area filtering
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1

        ttk.Label(parent, text="Area Filtering", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        row = self._add_slider(parent, row, "Min Area", self.params['min_area'], 0, 10000)
        row = self._add_slider(parent, row, "Max Area", self.params['max_area'], 1000, 500000)

        # Inpainting section
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1

        ttk.Label(parent, text="Inpainting", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        ttk.Checkbutton(parent, text="Enable Inpainting",
                       variable=self.params['enable_inpaint'],
                       command=self.update_mask).grid(row=row, column=0, columnspan=2,
                                                      sticky=tk.W, padx=20, pady=3)
        row += 1

        # Inpainting method selection
        ttk.Label(parent, text="Method:", font=('Arial', 9)).grid(
            row=row, column=0, sticky=tk.W, padx=20, pady=2)

        inpaint_combo = ttk.Combobox(parent, textvariable=self.params['inpaint_method'],
                                     values=['telea', 'ns', 'bilateral', 'morphological',
                                            'multiscale', 'patchmatch'], state='readonly', width=12)
        inpaint_combo.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        row = self._add_slider(parent, row, "Radius", self.params['inpaint_radius'], 1, 50)

        # PatchMatch parameters (shown when patchmatch is selected)
        self.patchmatch_frame = ttk.LabelFrame(parent, text="PatchMatch Parameters", padding="5")
        self.patchmatch_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        row += 1

        pm_row = 0
        pm_row = self._add_slider(self.patchmatch_frame, pm_row, "Patch Size",
                                  self.params['patchmatch_patch_size'], 3, 15, step=2)
        pm_row = self._add_slider(self.patchmatch_frame, pm_row, "Iterations",
                                  self.params['patchmatch_iterations'], 1, 10)
        pm_row = self._add_slider(self.patchmatch_frame, pm_row, "Guidance (α)",
                                  self.params['patchmatch_alpha'], 0, 1, step=0.1)

        # Action buttons
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1

        ttk.Button(parent, text="🔍 Preview Mask Only",
                  command=self.update_mask).grid(row=row, column=0, columnspan=2,
                                                 sticky=tk.EW, padx=5, pady=5)
        row += 1

        ttk.Button(parent, text="✨ Apply Filter",
                  command=self.process_image,
                  style='Accent.TButton').grid(row=row, column=0, columnspan=2,
                                                   sticky=tk.EW, padx=5, pady=5)

    def populate_method_params(self):
        """Populate method-specific parameters"""
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

        elif method == 'saturation':
            row = self._add_slider(self.method_params_frame, row, "Threshold",
                                  self.params['saturation_threshold'], 0, 255)

        elif method == 'edge':
            row = self._add_slider(self.method_params_frame, row, "Canny Low",
                                  self.params['canny_low'], 0, 500)
            row = self._add_slider(self.method_params_frame, row, "Canny High",
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
                     font=('Arial', 9, 'italic')).grid(row=row, column=0, columnspan=2, pady=5)
            row += 1
            row = self._add_slider(self.method_params_frame, row, "Markers",
                                  self.params['watershed_markers'], 2, 20)
            row = self._add_slider(self.method_params_frame, row, "Compactness",
                                  self.params['watershed_compactness'], 0.0001, 0.01, step=0.0001)

        elif method == 'dft':
            ttk.Label(self.method_params_frame, text="🆕 Fourier Transform Filter",
                     font=('Arial', 9, 'italic')).grid(row=row, column=0, columnspan=2, pady=5)
            row += 1

            ttk.Label(self.method_params_frame, text="Filter Type:").grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=2)
            dft_combo = ttk.Combobox(self.method_params_frame, textvariable=self.params['dft_filter_type'],
                                    values=['notch', 'highpass', 'bandreject', 'adaptive'],
                                    state='readonly', width=12)
            dft_combo.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            row += 1

            row = self._add_slider(self.method_params_frame, row, "Cutoff Freq",
                                  self.params['dft_cutoff'], 10, 100)
            row = self._add_slider(self.method_params_frame, row, "Radius",
                                  self.params['dft_radius'], 5, 50)

    def on_method_change(self):
        """Handle detection method change"""
        self.populate_method_params()
        self.update_mask()

    def _add_slider(self, parent, row, label, variable, from_, to, step=1):
        """Add a labeled slider"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)

        label_w = ttk.Label(frame, text=label, width=12)
        label_w.pack(side=tk.LEFT)

        value_label = ttk.Label(frame, text=str(variable.get()), width=6)
        value_label.pack(side=tk.RIGHT)

        def update_label(val):
            if step < 1:
                value_label.config(text=f"{float(val):.4f}")
            else:
                value_label.config(text=str(int(float(val))))
            if self.processor.original_image is not None and not self.processing:
                self.root.after(100, self.update_mask)

        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                          variable=variable, command=update_label)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        return row + 1

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )

        if file_path:
            self.current_image_path = file_path
            image = self.processor.load_image(file_path)
            self.display_image(image, self.original_canvas)
            self.display_image(image, self.working_canvas)
            self.status_var.set(f"✓ Loaded: {os.path.basename(file_path)}")
            self.update_history_ui()
            self.update_mask()

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
        """Display mask preview (called from main thread)"""
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

        if filter_type == 'notch':
            return DFTFilter.notch_filter(image, radius=radius, auto_detect=True)
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
                            from ..inpainting.basic import BasicInpainting
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
        """Display result (called from main thread)"""
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

    def get_current_params(self):
        """Get current parameter values"""
        return {k: v.get() for k, v in self.params.items()}

    def display_image(self, image, canvas):
        """Display an image on a canvas"""
        height, width = image.shape[:2]
        max_size = 380

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
            image_bgr = cv2.cvtColor(self.processor.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, image_bgr)
            self.status_var.set(f"✓ Saved: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Image saved successfully!")


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
