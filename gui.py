"""
PhotoRefine: GUI Module
User interface for interactive blob removal with multi-pass support
"""

import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import threading
from image_processor import BlobRemovalProcessor


class PhotoRefineGUI:
    """GUI with multi-pass filtering support and history management"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PhotoRefine - Interactive Blob Removal")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        self.processor = BlobRemovalProcessor()
        self.current_image_path = None
        self.processing = False
        
        # Parameters with expanded ranges
        self.params = {
            'detection_method': tk.StringVar(value='hsv'),
            'h_min': tk.IntVar(value=15),
            'h_max': tk.IntVar(value=45),
            's_min': tk.IntVar(value=80),
            's_max': tk.IntVar(value=255),
            'v_min': tk.IntVar(value=100),
            'v_max': tk.IntVar(value=255),
            'brightness_threshold': tk.IntVar(value=200),
            'saturation_threshold': tk.IntVar(value=100),
            'canny_low': tk.IntVar(value=50),
            'canny_high': tk.IntVar(value=150),
            'block_size': tk.IntVar(value=11),
            'c_value': tk.IntVar(value=2),
            'n_clusters': tk.IntVar(value=5),
            'morph_operation': tk.StringVar(value='close'),
            'kernel_size': tk.IntVar(value=5),
            'min_area': tk.IntVar(value=100),
            'max_area': tk.IntVar(value=50000),
            'inpaint_radius': tk.IntVar(value=5),
            'enable_inpaint': tk.BooleanVar(value=True),
            'use_working_image': tk.BooleanVar(value=True),
            # Advanced morphology
            'morph_iterations': tk.IntVar(value=1),
            # Gaussian blur
            'gaussian_blur': tk.IntVar(value=0),
            # Additional filters
            'median_blur': tk.IntVar(value=0),
            'bilateral_filter': tk.IntVar(value=0)
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
        canvas.configure(yscrollcommand=scrollbar.set, width=320)
        
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
        
        self.undo_btn = ttk.Button(button_frame, text="⬅ Undo", 
                                    command=self.undo_step, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=2)
        
        self.redo_btn = ttk.Button(button_frame, text="➡ Redo", 
                                    command=self.redo_step, state=tk.DISABLED)
        self.redo_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="🔄 Reset to Original", 
                  command=self.reset_to_original).pack(side=tk.LEFT, padx=2)
        
        self.history_label = ttk.Label(button_frame, text="No history", 
                                       font=('Arial', 9))
        self.history_label.pack(side=tk.RIGHT, padx=5)
    
    def _build_controls(self, parent):
        """Build control panel"""
        row = 0
        
        # Title
        ttk.Label(parent, text="PhotoRefine", font=('Arial', 14, 'bold')).grid(
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
        
        # Multi-pass mode
        ttk.Label(parent, text="Processing Mode", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        mode_frame = ttk.Frame(parent)
        mode_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        
        ttk.Radiobutton(mode_frame, text="Process from Original", 
                       variable=self.params['use_working_image'],
                       value=False).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(mode_frame, text="Process from Current", 
                       variable=self.params['use_working_image'],
                       value=True).pack(side=tk.LEFT, padx=3)
        row += 1
        
        # Info label
        info_label = ttk.Label(parent, 
            text="💡 'Process from Current' allows\nchaining multiple filters",
            font=('Arial', 8), foreground='gray')
        info_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
        row += 1
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # Detection method
        ttk.Label(parent, text="Detection Method", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        methods = [('HSV Color', 'hsv'), ('Brightness', 'brightness'), 
                   ('Saturation', 'saturation'), ('Edge', 'edge'), 
                   ('Adaptive', 'adaptive'), ('K-Means', 'kmeans')]
        
        for label, value in methods:
            ttk.Radiobutton(parent, text=label, variable=self.params['detection_method'], 
                           value=value, command=self.update_mask).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
            row += 1
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # HSV Parameters - Full range for precise control
        ttk.Label(parent, text="HSV Thresholds", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        row = self._add_slider(parent, row, "Hue Min", self.params['h_min'], 0, 179)
        row = self._add_slider(parent, row, "Hue Max", self.params['h_max'], 0, 179)
        row = self._add_slider(parent, row, "Sat Min", self.params['s_min'], 0, 255)
        row = self._add_slider(parent, row, "Sat Max", self.params['s_max'], 0, 255)
        row = self._add_slider(parent, row, "Val Min", self.params['v_min'], 0, 255)
        row = self._add_slider(parent, row, "Val Max", self.params['v_max'], 0, 255)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # Other parameters - Expanded ranges
        ttk.Label(parent, text="Other Thresholds", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        row = self._add_slider(parent, row, "Brightness", self.params['brightness_threshold'], 0, 255)
        row = self._add_slider(parent, row, "Saturation", self.params['saturation_threshold'], 0, 255)
        row = self._add_slider(parent, row, "Canny Low", self.params['canny_low'], 0, 500)
        row = self._add_slider(parent, row, "Canny High", self.params['canny_high'], 0, 500)
        row = self._add_slider(parent, row, "Block Size", self.params['block_size'], 3, 99, step=2)
        row = self._add_slider(parent, row, "C Value", self.params['c_value'], -20, 20)
        row = self._add_slider(parent, row, "Clusters", self.params['n_clusters'], 2, 20)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # Morphological operations
        ttk.Label(parent, text="Morphology", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        morph_frame = ttk.Frame(parent)
        morph_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        
        for i, (label, value) in enumerate([('None', 'none'), ('Open', 'open'), 
                                            ('Close', 'close'), ('Dilate', 'dilate')]):
            ttk.Radiobutton(morph_frame, text=label, variable=self.params['morph_operation'],
                           value=value, command=self.update_mask).pack(side=tk.LEFT, padx=3)
        row += 1
        
        row = self._add_slider(parent, row, "Kernel Size", self.params['kernel_size'], 1, 51, step=2)
        row = self._add_slider(parent, row, "Iterations", self.params['morph_iterations'], 1, 10)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # Area filtering - Much wider range
        ttk.Label(parent, text="Area Filtering", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        row = self._add_slider(parent, row, "Min Area", self.params['min_area'], 0, 10000)
        row = self._add_slider(parent, row, "Max Area", self.params['max_area'], 1000, 500000)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # Inpainting - Extended range
        ttk.Label(parent, text="Inpainting", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        ttk.Checkbutton(parent, text="Enable Inpainting", 
                       variable=self.params['enable_inpaint'],
                       command=self.update_mask).grid(row=row, column=0, columnspan=2, 
                                                      sticky=tk.W, padx=20, pady=3)
        row += 1
        
        row = self._add_slider(parent, row, "Radius", self.params['inpaint_radius'], 1, 50)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # Pre-processing filters
        ttk.Label(parent, text="Pre-Processing Filters", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        info_label2 = ttk.Label(parent, 
            text="💡 Apply these before detection",
            font=('Arial', 8), foreground='gray')
        info_label2.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
        row += 1
        
        row = self._add_slider(parent, row, "Gaussian Blur", self.params['gaussian_blur'], 0, 31, step=2)
        row = self._add_slider(parent, row, "Median Blur", self.params['median_blur'], 0, 31, step=2)
        row = self._add_slider(parent, row, "Bilateral Filter", self.params['bilateral_filter'], 0, 20)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                        sticky=tk.EW, pady=8, padx=5)
        row += 1
        
        # Action buttons
        ttk.Button(parent, text="🔍 Preview Mask Only", 
                  command=self.update_mask).grid(row=row, column=0, columnspan=2, 
                                                 sticky=tk.EW, padx=5, pady=5)
        row += 1
        
        ttk.Button(parent, text="✨ Apply Filter", 
                  command=self.process_image, 
                  style='Accent.TButton').grid(row=row, column=0, columnspan=2, 
                                                   sticky=tk.EW, padx=5, pady=5)
    
    def _add_slider(self, parent, row, label, variable, from_, to, step=1):
        """Add a labeled slider"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        
        label_w = ttk.Label(frame, text=label, width=12)
        label_w.pack(side=tk.LEFT)
        
        value_label = ttk.Label(frame, text=str(variable.get()), width=5)
        value_label.pack(side=tk.RIGHT)
        
        def update_label(val):
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
            params = self.get_current_params()
            params_copy = params.copy()
            params_copy['enable_inpaint'] = False
            use_working = params['use_working_image']
            
            _, mask = self.processor.process_with_params(
                params_copy, 
                use_working_image=use_working, 
                add_to_history=False
            )
            
            self.root.after(0, lambda: self._display_mask(mask))
        
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
    
    def process_image(self):
        """Process the image and apply inpainting"""
        if self.processor.original_image is None or self.processing:
            return
        
        self.processing = True
        mode = "current result" if self.params['use_working_image'].get() else "original"
        self.status_var.set(f"⏳ Applying filter to {mode}...")
        self.root.update()
        
        def process():
            params = self.get_current_params()
            use_working = params['use_working_image']
            
            result, mask = self.processor.process_with_params(
                params, 
                use_working_image=use_working,
                add_to_history=True
            )
            
            self.root.after(0, lambda: self._display_result(result, mask))
        
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
        return {
            'detection_method': self.params['detection_method'].get(),
            'h_min': self.params['h_min'].get(),
            'h_max': self.params['h_max'].get(),
            's_min': self.params['s_min'].get(),
            's_max': self.params['s_max'].get(),
            'v_min': self.params['v_min'].get(),
            'v_max': self.params['v_max'].get(),
            'brightness_threshold': self.params['brightness_threshold'].get(),
            'saturation_threshold': self.params['saturation_threshold'].get(),
            'canny_low': self.params['canny_low'].get(),
            'canny_high': self.params['canny_high'].get(),
            'block_size': self.params['block_size'].get(),
            'c_value': self.params['c_value'].get(),
            'n_clusters': self.params['n_clusters'].get(),
            'morph_operation': self.params['morph_operation'].get(),
            'kernel_size': self.params['kernel_size'].get(),
            'morph_iterations': self.params['morph_iterations'].get(),
            'min_area': self.params['min_area'].get(),
            'max_area': self.params['max_area'].get(),
            'inpaint_radius': self.params['inpaint_radius'].get(),
            'enable_inpaint': self.params['enable_inpaint'].get(),
            'use_working_image': self.params['use_working_image'].get(),
            'gaussian_blur': self.params['gaussian_blur'].get(),
            'median_blur': self.params['median_blur'].get(),
            'bilateral_filter': self.params['bilateral_filter'].get()
        }
    
    def display_image(self, image, canvas):
        """Display an image on a canvas"""
        height, width = image.shape[:2]
        max_size = 320
        
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