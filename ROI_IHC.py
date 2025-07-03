import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import os
import numpy as np
import cv2
import json
from skimage import color, segmentation, measure
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class IHC_ROI_Analyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("IHC ROI Analysis Tool v1.0")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.image = None
        self.photo = None
        self.canvas_image = None
        self.roi_rects = []
        self.roi_width = tk.IntVar(value=200)
        self.roi_height = tk.IntVar(value=200)
        self.num_rois = tk.IntVar(value=5)
        self.save_path = tk.StringVar(value="./ihc_analysis_results")
        self.scale_factor = 1.0
        self.zoom_level = 1.0
        
        # IHC-specific variables
        self.ihc_analysis_enabled = tk.BooleanVar(value=True)
        self.stain_type = tk.StringVar(value="DAB")  # DAB, H&E, Sirius Red
        self.analysis_mode = tk.StringVar(value="intensity")  # intensity, area, nuclei_count
        self.threshold_dab = tk.IntVar(value=150)
        self.threshold_hematoxylin = tk.IntVar(value=200)
        self.min_nucleus_size = tk.IntVar(value=50)
        self.max_nucleus_size = tk.IntVar(value=500)
        
        # Analysis results storage
        self.analysis_results = []
        self.color_deconvolved_images = {}
        
        # Interaction modes
        self.interaction_mode = tk.StringVar(value="create")
        self.show_grid = tk.BooleanVar(value=False)
        self.show_roi_info = tk.BooleanVar(value=True)
        self.show_analysis_overlay = tk.BooleanVar(value=False)
        
        # Action history for undo/redo
        self.action_history = []
        self.history_pointer = -1
        
        self.setup_ui()
        
        # Keyboard bindings
        self.root.bind('<Delete>', self.delete_selected_roi)
        self.root.bind('<BackSpace>', self.delete_selected_roi)
        self.root.bind('<Control-z>', self.undo_action)
        self.root.bind('<Control-y>', self.redo_action)
        self.root.bind('<Control-d>', lambda e: self.duplicate_selected_roi())
        self.root.bind('<F5>', lambda e: self.run_ihc_analysis())
        self.root.focus_set()
        
    def setup_ui(self):
        # Menu bar
        self.create_menu_bar()
        
        # Main container with three panels
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - ROI controls
        left_panel = ttk.Frame(main_container, width=280)
        main_container.add(left_panel, weight=0)
        
        # Middle panel - Image canvas
        middle_panel = ttk.Frame(main_container)
        main_container.add(middle_panel, weight=2)
        
        # Right panel - Analysis results
        right_panel = ttk.Frame(main_container, width=300)
        main_container.add(right_panel, weight=1)
        
        # Setup panels
        self.setup_left_panel(left_panel)
        self.setup_middle_panel(middle_panel)
        self.setup_right_panel(right_panel)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Open an IHC image to start analysis")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Drawing state
        self.drawing = False
        self.moving_roi = False
        self.selected_roi = None
        self.selected_roi_index = -1
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.move_offset_x = 0
        self.move_offset_y = 0
        self.grid_lines = []
        
    def create_menu_bar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open IHC Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save ROI Template", command=self.save_roi_template)
        file_menu.add_command(label="Load ROI Template", command=self.load_roi_template)
        file_menu.add_separator()
        file_menu.add_command(label="Export Analysis Results", command=self.export_analysis_results)
        file_menu.add_command(label="Export ROI Images", command=self.save_rois)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run IHC Analysis", command=self.run_ihc_analysis, accelerator="F5")
        analysis_menu.add_command(label="Color Deconvolution", command=self.color_deconvolution)
        analysis_menu.add_command(label="Nuclei Detection", command=self.detect_nuclei)
        analysis_menu.add_command(label="Staining Quantification", command=self.quantify_staining)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Clear Analysis Results", command=self.clear_analysis_results)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Grid", variable=self.show_grid, command=self.toggle_grid)
        view_menu.add_checkbutton(label="Show ROI Info", variable=self.show_roi_info, command=self.refresh_display)
        view_menu.add_checkbutton(label="Show Analysis Overlay", variable=self.show_analysis_overlay, command=self.refresh_display)
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=lambda: self.zoom(1.2))
        view_menu.add_command(label="Zoom Out", command=lambda: self.zoom(0.8))
        view_menu.add_command(label="Reset Zoom", command=lambda: self.zoom(1.0, reset=True))
        
    def setup_left_panel(self, panel):
        # ROI Controls Section
        roi_frame = ttk.LabelFrame(panel, text="ROI Controls", padding=10)
        roi_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Interaction mode
        mode_frame = ttk.Frame(roi_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(mode_frame, text="Mode:").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Create ROIs", variable=self.interaction_mode, 
                       value="create", command=self.change_interaction_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Select/Move", variable=self.interaction_mode, 
                       value="select", command=self.change_interaction_mode).pack(anchor=tk.W)
        
        # ROI settings
        settings_frame = ttk.Frame(roi_frame)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.roi_width, width=10).grid(row=0, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(settings_frame, text="Height:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.roi_height, width=10).grid(row=1, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(settings_frame, text="Max ROIs:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(settings_frame, from_=1, to=20, textvariable=self.num_rois, width=10).grid(row=2, column=1, padx=(5, 0), pady=2)
        
        # ROI actions
        actions_frame = ttk.Frame(roi_frame)
        actions_frame.pack(fill=tk.X)
        ttk.Button(actions_frame, text="Clear All ROIs", command=self.clear_rois).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Delete Selected", command=self.delete_selected_roi).pack(fill=tk.X, pady=2)
        
        # IHC Analysis Section
        ihc_frame = ttk.LabelFrame(panel, text="IHC Analysis Settings", padding=10)
        ihc_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(ihc_frame, text="Enable IHC Analysis", variable=self.ihc_analysis_enabled).pack(anchor=tk.W, pady=(0, 5))
        
        # Stain type
        ttk.Label(ihc_frame, text="Stain Type:").pack(anchor=tk.W)
        stain_combo = ttk.Combobox(ihc_frame, textvariable=self.stain_type, 
                                  values=["DAB", "H&E", "Sirius Red", "Custom"], state="readonly")
        stain_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Analysis mode
        ttk.Label(ihc_frame, text="Analysis Mode:").pack(anchor=tk.W)
        mode_combo = ttk.Combobox(ihc_frame, textvariable=self.analysis_mode,
                                 values=["intensity", "area", "nuclei_count", "comprehensive"], state="readonly")
        mode_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Thresholds
        threshold_frame = ttk.Frame(ihc_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(threshold_frame, text="DAB Threshold:").grid(row=0, column=0, sticky=tk.W)
        dab_scale = ttk.Scale(threshold_frame, from_=0, to=255, variable=self.threshold_dab, orient=tk.HORIZONTAL)
        dab_scale.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0))
        ttk.Label(threshold_frame, textvariable=self.threshold_dab, width=4).grid(row=0, column=2, padx=(5, 0))
        
        ttk.Label(threshold_frame, text="H Threshold:").grid(row=1, column=0, sticky=tk.W)
        h_scale = ttk.Scale(threshold_frame, from_=0, to=255, variable=self.threshold_hematoxylin, orient=tk.HORIZONTAL)
        h_scale.grid(row=1, column=1, sticky=tk.EW, padx=(5, 0))
        ttk.Label(threshold_frame, textvariable=self.threshold_hematoxylin, width=4).grid(row=1, column=2, padx=(5, 0))
        
        threshold_frame.columnconfigure(1, weight=1)
        
        # Nuclei detection settings
        nuclei_frame = ttk.LabelFrame(ihc_frame, text="Nuclei Detection", padding=5)
        nuclei_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(nuclei_frame, text="Min Size:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(nuclei_frame, textvariable=self.min_nucleus_size, width=8).grid(row=0, column=1, padx=(5, 0))
        
        ttk.Label(nuclei_frame, text="Max Size:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(nuclei_frame, textvariable=self.max_nucleus_size, width=8).grid(row=1, column=1, padx=(5, 0))
        
        # Analysis buttons
        analysis_buttons_frame = ttk.Frame(ihc_frame)
        analysis_buttons_frame.pack(fill=tk.X)
        ttk.Button(analysis_buttons_frame, text="Run Analysis (F5)", command=self.run_ihc_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_buttons_frame, text="Color Deconvolution", command=self.color_deconvolution).pack(fill=tk.X, pady=2)
        
        # ROI List
        list_frame = ttk.LabelFrame(panel, text="ROI List", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.roi_listbox = tk.Listbox(list_container, height=8)
        roi_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.roi_listbox.yview)
        self.roi_listbox.configure(yscrollcommand=roi_scrollbar.set)
        
        self.roi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        roi_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.roi_listbox.bind('<<ListboxSelect>>', self.on_roi_list_select)
        
    def setup_middle_panel(self, panel):
        # Toolbar
        toolbar_frame = ttk.Frame(panel)
        toolbar_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(toolbar_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="Zoom In", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="Reset Zoom", command=lambda: self.zoom(1.0, reset=True)).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Checkbutton(toolbar_frame, text="Grid", variable=self.show_grid, command=self.toggle_grid).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(toolbar_frame, text="Analysis Overlay", variable=self.show_analysis_overlay, command=self.refresh_display).pack(side=tk.LEFT)
        
        # Image canvas
        canvas_frame = ttk.LabelFrame(panel, text="IHC Image Canvas", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', cursor='crosshair')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        
    def setup_right_panel(self, panel):
        # Analysis Results Section
        results_frame = ttk.LabelFrame(panel, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results table
        columns = ("ROI", "Metric", "Value")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=80)
        
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Summary statistics
        summary_frame = ttk.LabelFrame(panel, text="Summary Statistics", padding=10)
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.summary_text = tk.Text(summary_frame, height=8, wrap=tk.WORD)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export buttons
        export_frame = ttk.Frame(panel)
        export_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(export_frame, text="Export Results", command=self.export_analysis_results).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save Report", command=self.save_analysis_report).pack(fill=tk.X, pady=2)
        
    # ROI Management Methods (from original code)
    def change_interaction_mode(self):
        mode = self.interaction_mode.get()
        if mode == "create":
            self.canvas.configure(cursor='crosshair')
            self.deselect_all_rois()
        else:
            self.canvas.configure(cursor='hand2')
        self.update_status()
        
    def on_canvas_click(self, event):
        if not self.image:
            return
            
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        mode = self.interaction_mode.get()
        
        if mode == "create":
            if len(self.roi_rects) >= self.num_rois.get():
                messagebox.showwarning("Limit Reached", f"Maximum {self.num_rois.get()} ROIs allowed.")
                return
                
            self.drawing = True
            self.start_x = x
            self.start_y = y
            
            width = self.roi_width.get() * self.scale_factor * self.zoom_level
            height = self.roi_height.get() * self.scale_factor * self.zoom_level
            
            self.current_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y,
                self.start_x + width, self.start_y + height,
                outline='red', width=2, fill='', stipple='gray25'
            )
            
        elif mode == "select":
            clicked_roi_index = self.find_roi_at_position(x, y)
            if clicked_roi_index is not None:
                self.select_roi(clicked_roi_index)
                self.moving_roi = True
                self.start_x = x
                self.start_y = y
                roi_coords = self.canvas.coords(self.selected_roi['rect_id'])
                self.move_offset_x = x - roi_coords[0]
                self.move_offset_y = y - roi_coords[1]
            else:
                self.deselect_all_rois()
                
    def on_canvas_drag(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        if self.drawing and self.current_rect:
            width = self.roi_width.get() * self.scale_factor * self.zoom_level
            height = self.roi_height.get() * self.scale_factor * self.zoom_level
            self.canvas.coords(self.current_rect, x, y, x + width, y + height)
            
        elif self.moving_roi and self.selected_roi:
            new_x = x - self.move_offset_x
            new_y = y - self.move_offset_y
            width = self.roi_width.get() * self.scale_factor * self.zoom_level
            height = self.roi_height.get() * self.scale_factor * self.zoom_level
            
            max_x = (self.image.width * self.scale_factor * self.zoom_level) - width
            max_y = (self.image.height * self.scale_factor * self.zoom_level) - height
            
            new_x = max(0, min(new_x, max_x))
            new_y = max(0, min(new_y, max_y))
            
            self.canvas.coords(self.selected_roi['rect_id'], 
                             new_x, new_y, new_x + width, new_y + height)
            
    def on_canvas_release(self, event):
        if self.drawing and self.current_rect:
            self.finish_creating_roi()
        elif self.moving_roi and self.selected_roi:
            self.finish_moving_roi()
            
    def finish_creating_roi(self):
        self.drawing = False
        coords = self.canvas.coords(self.current_rect)
        original_coords = self.display_to_original_coords(coords)
        
        roi_info = {
            'rect_id': self.current_rect,
            'coords': original_coords,
            'display_coords': coords,
            'label': f"ROI_{len(self.roi_rects) + 1:02d}",
            'analysis_results': {}
        }
        
        self.save_action_state()
        self.roi_rects.append(roi_info)
        self.current_rect = None
        self.update_roi_list()
        self.update_status()
        
    def finish_moving_roi(self):
        self.moving_roi = False
        coords = self.canvas.coords(self.selected_roi['rect_id'])
        original_coords = self.display_to_original_coords(coords)
        self.selected_roi['coords'] = original_coords
        self.selected_roi['display_coords'] = coords
        self.update_roi_list()
        
    # IHC Analysis Methods
    def color_deconvolution(self):
        """Perform color deconvolution to separate DAB and Hematoxylin stains"""
        if not self.image:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
            
        try:
            # Convert PIL image to numpy array
            img_array = np.array(self.image)
            
            # H&E to RGB color deconvolution matrix (from Ruifrok & Johnston)
            if self.stain_type.get() == "DAB":
                # DAB-Hematoxylin matrix
                stain_matrix = np.array([
                    [0.650, 0.704, 0.286],  # Hematoxylin
                    [0.268, 0.570, 0.776],  # DAB
                    [0.0, 0.0, 0.0]         # Null vector
                ])
            else:
                # H&E matrix
                stain_matrix = np.array([
                    [0.644, 0.717, 0.267],  # Hematoxylin
                    [0.093, 0.954, 0.283],  # Eosin
                    [0.0, 0.0, 0.0]         # Null vector
                ])
            
            # Normalize image
            img_normalized = img_array.astype(np.float64) / 255.0
            img_normalized[img_normalized == 0] = 1e-6
            
            # Convert to optical density
            od = -np.log(img_normalized)
            
            # Deconvolve
            stains = np.dot(od.reshape(-1, 3), np.linalg.pinv(stain_matrix.T)).reshape(img_array.shape)
            
            # Convert back to RGB
            self.color_deconvolved_images = {}
            for i, stain_name in enumerate(['Hematoxylin', 'DAB' if self.stain_type.get() == "DAB" else 'Eosin']):
                stain_od = np.zeros_like(od)
                stain_od[:, :, :] = stains[:, :, i:i+1] * stain_matrix[i:i+1, :]
                stain_rgb = np.exp(-stain_od) * 255
                stain_rgb = np.clip(stain_rgb, 0, 255).astype(np.uint8)
                self.color_deconvolved_images[stain_name] = stain_rgb
                
            messagebox.showinfo("Success", "Color deconvolution completed successfully!")
            self.update_status("Color deconvolution completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Color deconvolution failed: {str(e)}")
            
    def detect_nuclei(self):
        """Detect nuclei in the hematoxylin channel"""
        if 'Hematoxylin' not in self.color_deconvolved_images:
            messagebox.showwarning("Warning", "Please run color deconvolution first.")
            return
            
        try:
            hematoxylin = self.color_deconvolved_images['Hematoxylin']
            gray = cv2.cvtColor(hematoxylin, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, self.threshold_hematoxylin.get(), 255, cv2.THRESH_BINARY_INV)
            
            # Remove noise
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter nuclei by size
            nuclei_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_nucleus_size.get() <= area <= self.max_nucleus_size.get():
                    nuclei_contours.append(contour)
            
            self.nuclei_contours = nuclei_contours
            messagebox.showinfo("Success", f"Detected {len(nuclei_contours)} nuclei")
            self.update_status(f"Detected {len(nuclei_contours)} nuclei")
            
        except Exception as e:
            messagebox.showerror("Error", f"Nuclei detection failed: {str(e)}")
            
    def quantify_staining(self):
        """Quantify staining intensity in ROIs"""
        if not self.roi_rects:
            messagebox.showwarning("Warning", "Please define ROIs first.")
            return
            
        if not self.color_deconvolved_images:
            messagebox.showwarning("Warning", "Please run color deconvolution first.")
            return
            
        try:
            img_array = np.array(self.image)
            dab_channel = self.color_deconvolved_images.get('DAB', self.color_deconvolved_images.get('Eosin'))
            
            self.analysis_results = []
            
            for i, roi in enumerate(self.roi_rects):
                coords = roi['coords']
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                
                # Extract ROI from original and deconvolved images
                roi_original = img_array[y1:y2, x1:x2]
                roi_dab = dab_channel[y1:y2, x1:x2] if dab_channel is not None else None
                
                results = {}
                
                # Calculate basic metrics
                if roi_dab is not None:
                    dab_gray = cv2.cvtColor(roi_dab, cv2.COLOR_RGB2GRAY)
                    
                    # Intensity measurements
                    results['mean_intensity'] = np.mean(dab_gray)
                    results['median_intensity'] = np.median(dab_gray)
                    results['std_intensity'] = np.std(dab_gray)
                    
                    # Positive area calculation
                    _, positive_mask = cv2.threshold(dab_gray, self.threshold_dab.get(), 255, cv2.THRESH_BINARY)
                    positive_pixels = np.sum(positive_mask > 0)
                    total_pixels = dab_gray.size
                    results['positive_area_percent'] = (positive_pixels / total_pixels) * 100
                    results['positive_pixel_count'] = positive_pixels
                    results['total_pixel_count'] = total_pixels
                    
                    # Optical density
                    roi_normalized = roi_original.astype(np.float64) / 255.0
                    roi_normalized[roi_normalized == 0] = 1e-6
                    od = -np.log(roi_normalized)
                    results['mean_optical_density'] = np.mean(od)
                
                # Nuclei count in ROI
                if hasattr(self, 'nuclei_contours'):
                    nuclei_in_roi = 0
                    for contour in self.nuclei_contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                nuclei_in_roi += 1
                    results['nuclei_count'] = nuclei_in_roi
                    
                    # H-score calculation (if nuclei detected)
                    if nuclei_in_roi > 0:
                        h_score = self.calculate_h_score(roi_dab, nuclei_in_roi)
                        results['h_score'] = h_score
                
                roi['analysis_results'] = results
                self.analysis_results.append({
                    'roi_id': roi['label'],
                    'results': results
                })
            
            self.update_results_display()
            messagebox.showinfo("Success", "Staining quantification completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Staining quantification failed: {str(e)}")
            
    def calculate_h_score(self, roi_dab, nuclei_count):
        """Calculate H-score for immunohistochemical staining"""
        if roi_dab is None:
            return 0
            
        gray = cv2.cvtColor(roi_dab, cv2.COLOR_RGB2GRAY)
        
        # Define intensity ranges (0: negative, 1: weak, 2: moderate, 3: strong)
        weak_threshold = self.threshold_dab.get() * 0.3
        moderate_threshold = self.threshold_dab.get() * 0.6
        strong_threshold = self.threshold_dab.get()
        
        # Count pixels in each category
        negative = np.sum(gray < weak_threshold)
        weak = np.sum((gray >= weak_threshold) & (gray < moderate_threshold))
        moderate = np.sum((gray >= moderate_threshold) & (gray < strong_threshold))
        strong = np.sum(gray >= strong_threshold)
        
        total_pixels = gray.size
        
        # Calculate percentages
        pct_negative = (negative / total_pixels) * 100
        pct_weak = (weak / total_pixels) * 100
        pct_moderate = (moderate / total_pixels) * 100
        pct_strong = (strong / total_pixels) * 100
        
        # H-score = (1 × %weak) + (2 × %moderate) + (3 × %strong)
        h_score = (1 * pct_weak) + (2 * pct_moderate) + (3 * pct_strong)
        
        return h_score
        
    def run_ihc_analysis(self):
        """Run comprehensive IHC analysis"""
        if not self.ihc_analysis_enabled.get():
            messagebox.showinfo("Info", "IHC analysis is disabled. Enable it in the settings.")
            return
            
        if not self.image:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
            
        if not self.roi_rects:
            messagebox.showwarning("Warning", "Please define ROIs first.")
            return
            
        try:
            # Show progress
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Running IHC Analysis...")
            progress_window.geometry("300x100")
            progress_window.transient(self.root)
            
            progress_label = ttk.Label(progress_window, text="Starting analysis...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, mode='determinate', maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            self.root.update()
            
            # Step 1: Color deconvolution
            progress_label.config(text="Performing color deconvolution...")
            progress_bar['value'] = 25
            self.root.update()
            self.color_deconvolution()
            
            # Step 2: Nuclei detection
            progress_label.config(text="Detecting nuclei...")
            progress_bar['value'] = 50
            self.root.update()
            self.detect_nuclei()
            
            # Step 3: Staining quantification
            progress_label.config(text="Quantifying staining...")
            progress_bar['value'] = 75
            self.root.update()
            self.quantify_staining()
            
            # Step 4: Generate summary
            progress_label.config(text="Generating summary...")
            progress_bar['value'] = 100
            self.root.update()
            self.generate_analysis_summary()
            
            progress_window.destroy()
            
            messagebox.showinfo("Success", "IHC analysis completed successfully!")
            self.update_status("IHC analysis completed")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"IHC analysis failed: {str(e)}")
            
    def update_results_display(self):
        """Update the results tree view"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Add new results
        for result in self.analysis_results:
            roi_id = result['roi_id']
            results = result['results']
            
            for metric, value in results.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                    
                self.results_tree.insert('', 'end', values=(roi_id, metric, formatted_value))
                
    def generate_analysis_summary(self):
        """Generate summary statistics"""
        if not self.analysis_results:
            return
            
        summary_text = "=== IHC ANALYSIS SUMMARY ===\n\n"
        summary_text += f"Image: {getattr(self, 'image_name', 'Unknown')}\n"
        summary_text += f"Stain Type: {self.stain_type.get()}\n"
        summary_text += f"Analysis Mode: {self.analysis_mode.get()}\n"
        summary_text += f"Number of ROIs: {len(self.analysis_results)}\n\n"
        
        # Calculate aggregate statistics
        all_intensities = []
        all_positive_areas = []
        all_nuclei_counts = []
        all_h_scores = []
        
        for result in self.analysis_results:
            results = result['results']
            if 'mean_intensity' in results:
                all_intensities.append(results['mean_intensity'])
            if 'positive_area_percent' in results:
                all_positive_areas.append(results['positive_area_percent'])
            if 'nuclei_count' in results:
                all_nuclei_counts.append(results['nuclei_count'])
            if 'h_score' in results:
                all_h_scores.append(results['h_score'])
        
        if all_intensities:
            summary_text += f"Mean Intensity:\n"
            summary_text += f"  Average: {np.mean(all_intensities):.2f}\n"
            summary_text += f"  Std Dev: {np.std(all_intensities):.2f}\n"
            summary_text += f"  Range: {np.min(all_intensities):.2f} - {np.max(all_intensities):.2f}\n\n"
        
        if all_positive_areas:
            summary_text += f"Positive Area (%):\n"
            summary_text += f"  Average: {np.mean(all_positive_areas):.2f}%\n"
            summary_text += f"  Std Dev: {np.std(all_positive_areas):.2f}%\n"
            summary_text += f"  Range: {np.min(all_positive_areas):.2f}% - {np.max(all_positive_areas):.2f}%\n\n"
        
        if all_nuclei_counts:
            summary_text += f"Nuclei Count:\n"
            summary_text += f"  Total: {np.sum(all_nuclei_counts)}\n"
            summary_text += f"  Average per ROI: {np.mean(all_nuclei_counts):.1f}\n"
            summary_text += f"  Range: {np.min(all_nuclei_counts)} - {np.max(all_nuclei_counts)}\n\n"
        
        if all_h_scores:
            summary_text += f"H-Score:\n"
            summary_text += f"  Average: {np.mean(all_h_scores):.2f}\n"
            summary_text += f"  Std Dev: {np.std(all_h_scores):.2f}\n"
            summary_text += f"  Range: {np.min(all_h_scores):.2f} - {np.max(all_h_scores):.2f}\n\n"
        
        # Classification based on H-score
        if all_h_scores:
            avg_h_score = np.mean(all_h_scores)
            if avg_h_score < 50:
                classification = "Weak positive"
            elif avg_h_score < 150:
                classification = "Moderate positive"
            else:
                classification = "Strong positive"
            summary_text += f"Overall Classification: {classification}\n"
        
        # Update summary text widget
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
        
    # File I/O Methods
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select IHC Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.image = Image.open(file_path)
                self.image_path = file_path
                self.image_name = os.path.splitext(os.path.basename(file_path))[0]
                self.zoom_level = 1.0
                self.display_image()
                self.clear_rois()
                self.clear_analysis_results()
                self.clear_action_history()
                self.status_var.set(f"Loaded: {os.path.basename(file_path)} ({self.image.width}x{self.image.height})")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")
                
    def save_roi_template(self):
        """Save ROI configuration as template"""
        if not self.roi_rects:
            messagebox.showwarning("Warning", "No ROIs to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save ROI Template",
            defaultextension=".json",
            filetypes=[("JSON Template", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                template_data = {
                    'roi_width': self.roi_width.get(),
                    'roi_height': self.roi_height.get(),
                    'stain_type': self.stain_type.get(),
                    'analysis_mode': self.analysis_mode.get(),
                    'threshold_dab': self.threshold_dab.get(),
                    'threshold_hematoxylin': self.threshold_hematoxylin.get(),
                    'min_nucleus_size': self.min_nucleus_size.get(),
                    'max_nucleus_size': self.max_nucleus_size.get(),
                    'rois': [{'coords': roi['coords'], 'label': roi['label']} for roi in self.roi_rects]
                }
                
                with open(file_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
                    
                messagebox.showinfo("Success", f"Template saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save template: {str(e)}")
                
    def load_roi_template(self):
        """Load ROI configuration from template"""
        file_path = filedialog.askopenfilename(
            title="Load ROI Template",
            filetypes=[("JSON Template", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    template_data = json.load(f)
                    
                # Clear existing ROIs
                self.clear_rois()
                
                # Load settings
                if 'roi_width' in template_data:
                    self.roi_width.set(template_data['roi_width'])
                if 'roi_height' in template_data:
                    self.roi_height.set(template_data['roi_height'])
                if 'stain_type' in template_data:
                    self.stain_type.set(template_data['stain_type'])
                if 'analysis_mode' in template_data:
                    self.analysis_mode.set(template_data['analysis_mode'])
                if 'threshold_dab' in template_data:
                    self.threshold_dab.set(template_data['threshold_dab'])
                if 'threshold_hematoxylin' in template_data:
                    self.threshold_hematoxylin.set(template_data['threshold_hematoxylin'])
                if 'min_nucleus_size' in template_data:
                    self.min_nucleus_size.set(template_data['min_nucleus_size'])
                if 'max_nucleus_size' in template_data:
                    self.max_nucleus_size.set(template_data['max_nucleus_size'])
                    
                # Load ROIs
                if self.image and 'rois' in template_data:
                    for roi_data in template_data['rois']:
                        coords = roi_data['coords']
                        display_coords = self.original_to_display_coords(coords)
                        
                        rect_id = self.canvas.create_rectangle(
                            *display_coords, outline='red', width=2, fill='', stipple='gray25'
                        )
                        
                        roi_info = {
                            'rect_id': rect_id,
                            'coords': coords,
                            'display_coords': display_coords,
                            'label': roi_data.get('label', f"ROI_{len(self.roi_rects) + 1:02d}"),
                            'analysis_results': {}
                        }
                        
                        self.roi_rects.append(roi_info)
                        
                    self.update_roi_list()
                    self.update_status()
                    
                messagebox.showinfo("Success", f"Template loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load template: {str(e)}")
                
    def export_analysis_results(self):
        """Export analysis results to CSV"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Analysis Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                import csv
                
                with open(file_path, 'w', newline='') as csvfile:
                    fieldnames = ['ROI_ID']
                    
                    # Get all possible metrics
                    all_metrics = set()
                    for result in self.analysis_results:
                        all_metrics.update(result['results'].keys())
                    
                    fieldnames.extend(sorted(all_metrics))
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in self.analysis_results:
                        row = {'ROI_ID': result['roi_id']}
                        row.update(result['results'])
                        writer.writerow(row)
                        
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                
    def save_analysis_report(self):
        """Save comprehensive analysis report"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # Write summary
                    f.write(self.summary_text.get(1.0, tk.END))
                    
                    # Write detailed results
                    f.write("\n\n=== DETAILED RESULTS ===\n\n")
                    for result in self.analysis_results:
                        f.write(f"ROI: {result['roi_id']}\n")
                        for metric, value in result['results'].items():
                            f.write(f"  {metric}: {value}\n")
                        f.write("\n")
                        
                messagebox.showinfo("Success", f"Report saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")
                
    # Utility Methods (from original ROI Cropper)
    def display_to_original_coords(self, display_coords):
        return [coord / (self.scale_factor * self.zoom_level) for coord in display_coords]
        
    def original_to_display_coords(self, original_coords):
        return [coord * self.scale_factor * self.zoom_level for coord in original_coords]
        
    def find_roi_at_position(self, x, y):
        for i, roi in enumerate(self.roi_rects):
            coords = self.canvas.coords(roi['rect_id'])
            if (coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]):
                return i
        return None
        
    def select_roi(self, index):
        if 0 <= index < len(self.roi_rects):
            self.deselect_all_rois()
            self.selected_roi = self.roi_rects[index]
            self.selected_roi_index = index
            self.canvas.itemconfig(self.selected_roi['rect_id'], outline='blue', width=3)
            self.roi_listbox.selection_clear(0, tk.END)
            self.roi_listbox.selection_set(index)
            self.update_status()
            
    def deselect_all_rois(self):
        for roi in self.roi_rects:
            self.canvas.itemconfig(roi['rect_id'], outline='red', width=2)
        self.selected_roi = None
        self.selected_roi_index = -1
        self.roi_listbox.selection_clear(0, tk.END)
        
    def delete_selected_roi(self, event=None):
        if self.selected_roi_index >= 0:
            self.delete_roi(self.selected_roi_index)
            
    def delete_roi(self, index):
        if 0 <= index < len(self.roi_rects):
            self.save_action_state()
            roi = self.roi_rects[index]
            self.canvas.delete(roi['rect_id'])
            self.roi_rects.pop(index)
            
            if self.selected_roi_index == index:
                self.selected_roi = None
                self.selected_roi_index = -1
            elif self.selected_roi_index > index:
                self.selected_roi_index -= 1
                
            self.update_roi_list()
            self.update_status()
            
    def duplicate_selected_roi(self, event=None):
        if self.selected_roi_index >= 0:
            self.duplicate_roi(self.selected_roi_index)
            
    def duplicate_roi(self, index):
        if 0 <= index < len(self.roi_rects) and len(self.roi_rects) < self.num_rois.get():
            self.save_action_state()
            original_roi = self.roi_rects[index]
            
            offset = 20
            new_coords = [
                original_roi['coords'][0] + offset,
                original_roi['coords'][1] + offset,
                original_roi['coords'][2] + offset,
                original_roi['coords'][3] + offset
            ]
            
            display_coords = self.original_to_display_coords(new_coords)
            new_rect = self.canvas.create_rectangle(
                *display_coords, outline='red', width=2, fill='', stipple='gray25'
            )
            
            new_roi = {
                'rect_id': new_rect,
                'coords': new_coords,
                'display_coords': display_coords,
                'label': f"ROI_{len(self.roi_rects) + 1:02d}",
                'analysis_results': {}
            }
            
            self.roi_rects.append(new_roi)
            self.update_roi_list()
            self.update_status()
            
    def clear_rois(self):
        for roi in self.roi_rects:
            self.canvas.delete(roi['rect_id'])
        self.roi_rects = []
        self.deselect_all_rois()
        self.clear_grid()
        self.update_roi_list()
        self.clear_analysis_results()
        self.update_status()
        
    def clear_analysis_results(self):
        self.analysis_results = []
        self.color_deconvolved_images = {}
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.summary_text.delete(1.0, tk.END)
        
    def save_rois(self):
        if not self.image or not self.roi_rects:
            messagebox.showwarning("Warning", "No image loaded or no ROIs defined.")
            return
            
        save_dir = self.save_path.get()
        if not save_dir:
            save_dir = filedialog.askdirectory(title="Select Save Directory")
            if not save_dir:
                return
            self.save_path.set(save_dir)
            
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            saved_count = 0
            for roi in self.roi_rects:
                coords = roi['coords']
                left, top, right, bottom = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                
                if right > left and bottom > top:
                    cropped = self.image.crop((left, top, right, bottom))
                    filename = f"{self.image_name}_{roi['label']}.png"
                    filepath = os.path.join(save_dir, filename)
                    cropped.save(filepath)
                    saved_count += 1
                    
            messagebox.showinfo("Success", f"Saved {saved_count} ROI image(s) to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save ROIs: {str(e)}")
            
    def display_image(self):
        if not self.image:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / self.image.width
            scale_y = canvas_height / self.image.height
            self.scale_factor = min(scale_x, scale_y, 1.0)
        else:
            self.scale_factor = 1.0
            
        display_width = int(self.image.width * self.scale_factor * self.zoom_level)
        display_height = int(self.image.height * self.scale_factor * self.zoom_level)
        
        display_image = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_image)
        
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        self.refresh_display()
        
    def refresh_display(self):
        if not self.image:
            return
            
        for roi in self.roi_rects:
            display_coords = self.original_to_display_coords(roi['coords'])
            roi['display_coords'] = display_coords
            self.canvas.coords(roi['rect_id'], *display_coords)
            
        if self.show_grid.get():
            self.draw_grid()
            
        if self.show_roi_info.get():
            self.draw_roi_info()
            
        if self.show_analysis_overlay.get():
            self.draw_analysis_overlay()
            
    def draw_roi_info(self):
        for roi in self.roi_rects:
            coords = roi['display_coords']
            text_x = coords[0] + 5
            text_y = coords[1] + 5
            
            info_text = f"{roi['label']}\n{int(roi['coords'][2]-roi['coords'][0])}x{int(roi['coords'][3]-roi['coords'][1])}"
            
            # Add analysis results if available
            if roi['analysis_results']:
                if 'positive_area_percent' in roi['analysis_results']:
                    info_text += f"\nPos: {roi['analysis_results']['positive_area_percent']:.1f}%"
                if 'h_score' in roi['analysis_results']:
                    info_text += f"\nH: {roi['analysis_results']['h_score']:.1f}"
            
            text_id = self.canvas.create_text(text_x, text_y, text=info_text, 
                                            anchor=tk.NW, fill='white', font=('Arial', 8))
            bbox = self.canvas.bbox(text_id)
            if bbox:
                bg_id = self.canvas.create_rectangle(bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2, 
                                                   fill='black', outline='')
                self.canvas.tag_lower(bg_id, text_id)
                
    def draw_analysis_overlay(self):
        """Draw analysis overlay showing positive areas"""
        if not hasattr(self, 'nuclei_contours') and not self.color_deconvolved_images:
            return
            
        # Draw nuclei contours if available
        if hasattr(self, 'nuclei_contours'):
            for contour in self.nuclei_contours:
                # Scale contour coordinates
                scaled_contour = contour * self.scale_factor * self.zoom_level
                points = scaled_contour.reshape(-1, 2).astype(int)
                
                # Draw contour
                for i in range(len(points)):
                    start = tuple(points[i])
                    end = tuple(points[(i + 1) % len(points)])
                    self.canvas.create_line(start[0], start[1], end[0], end[1], 
                                          fill='cyan', width=1, tags='analysis_overlay')
                                          
    def draw_grid(self):
        if not self.image:
            return
            
        self.clear_grid()
        
        grid_size = 50 * self.scale_factor * self.zoom_level
        width = self.image.width * self.scale_factor * self.zoom_level
        height = self.image.height * self.scale_factor * self.zoom_level
        
        x = grid_size
        while x < width:
            line = self.canvas.create_line(x, 0, x, height, fill='gray', dash=(2, 2), tags='grid')
            self.grid_lines.append(line)
            x += grid_size
            
        y = grid_size
        while y < height:
            line = self.canvas.create_line(0, y, width, y, fill='gray', dash=(2, 2), tags='grid')
            self.grid_lines.append(line)
            y += grid_size
            
    def clear_grid(self):
        for line in self.grid_lines:
            self.canvas.delete(line)
        self.grid_lines = []
        
    def toggle_grid(self):
        if self.show_grid.get():
            self.draw_grid()
        else:
            self.clear_grid()
            
    def zoom(self, factor, reset=False):
        if not self.image:
            return
            
        if reset:
            self.zoom_level = 1.0
        else:
            new_zoom = self.zoom_level * factor
            self.zoom_level = max(0.1, min(new_zoom, 5.0))
            
        self.display_image()
        self.refresh_display()
        self.update_status()
        
    def on_right_click(self, event):
        if not self.image:
            return
            
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        clicked_roi_index = self.find_roi_at_position(x, y)
        
        if clicked_roi_index is not None:
            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(label="Select ROI", command=lambda: self.select_roi(clicked_roi_index))
            context_menu.add_command(label="Delete ROI", command=lambda: self.delete_roi(clicked_roi_index))
            context_menu.add_command(label="Duplicate ROI", command=lambda: self.duplicate_roi(clicked_roi_index))
            context_menu.add_separator()
            context_menu.add_command(label="Analyze This ROI", command=lambda: self.analyze_single_roi(clicked_roi_index))
            context_menu.add_command(label="View ROI Results", command=lambda: self.show_roi_results(clicked_roi_index))
            
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
                
    def analyze_single_roi(self, roi_index):
        """Analyze a single ROI"""
        if not self.color_deconvolved_images:
            messagebox.showwarning("Warning", "Please run color deconvolution first.")
            return
            
        try:
            roi = self.roi_rects[roi_index]
            coords = roi['coords']
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            
            # Extract ROI and analyze
            img_array = np.array(self.image)
            roi_original = img_array[y1:y2, x1:x2]
            
            # Perform analysis for this ROI only
            dab_channel = self.color_deconvolved_images.get('DAB', self.color_deconvolved_images.get('Eosin'))
            if dab_channel is not None:
                roi_dab = dab_channel[y1:y2, x1:x2]
                dab_gray = cv2.cvtColor(roi_dab, cv2.COLOR_RGB2GRAY)
                
                results = {}
                results['mean_intensity'] = np.mean(dab_gray)
                results['median_intensity'] = np.median(dab_gray)
                
                _, positive_mask = cv2.threshold(dab_gray, self.threshold_dab.get(), 255, cv2.THRESH_BINARY)
                positive_pixels = np.sum(positive_mask > 0)
                total_pixels = dab_gray.size
                results['positive_area_percent'] = (positive_pixels / total_pixels) * 100
                
                if hasattr(self, 'nuclei_contours'):
                    nuclei_in_roi = 0
                    for contour in self.nuclei_contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                nuclei_in_roi += 1
                    results['nuclei_count'] = nuclei_in_roi
                    
                    if nuclei_in_roi > 0:
                        h_score = self.calculate_h_score(roi_dab, nuclei_in_roi)
                        results['h_score'] = h_score
                
                roi['analysis_results'] = results
                
                # Update displays
                self.update_roi_list()
                self.refresh_display()
                
                messagebox.showinfo("Analysis Complete", 
                                  f"ROI {roi['label']} analysis completed!\n"
                                  f"Positive Area: {results['positive_area_percent']:.1f}%\n"
                                  f"Mean Intensity: {results['mean_intensity']:.1f}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Single ROI analysis failed: {str(e)}")
            
    def show_roi_results(self, roi_index):
        """Show detailed results for a specific ROI"""
        roi = self.roi_rects[roi_index]
        if not roi['analysis_results']:
            messagebox.showinfo("No Results", f"No analysis results available for {roi['label']}")
            return
            
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Results - {roi['label']}")
        results_window.geometry("400x300")
        results_window.transient(self.root)
        
        # Create scrollable text widget
        text_frame = ttk.Frame(results_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Format results
        results_text = f"=== RESULTS FOR {roi['label']} ===\n\n"
        coords = roi['coords']
        results_text += f"Position: ({int(coords[0])}, {int(coords[1])})\n"
        results_text += f"Size: {int(coords[2]-coords[0])} x {int(coords[3]-coords[1])} pixels\n"
        results_text += f"Area: {int((coords[2]-coords[0]) * (coords[3]-coords[1]))} pixels²\n\n"
        
        results_text += "ANALYSIS RESULTS:\n"
        for metric, value in roi['analysis_results'].items():
            if isinstance(value, float):
                results_text += f"{metric}: {value:.3f}\n"
            else:
                results_text += f"{metric}: {value}\n"
        
        text_widget.insert(1.0, results_text)
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(results_window, text="Close", command=results_window.destroy).pack(pady=10)
        
    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom(1.1)
        else:
            self.zoom(0.9)
            
    def on_roi_list_select(self, event):
        selection = self.roi_listbox.curselection()
        if selection:
            self.select_roi(selection[0])
            
    def update_roi_list(self):
        self.roi_listbox.delete(0, tk.END)
        for i, roi in enumerate(self.roi_rects):
            coords = roi['coords']
            info = f"{roi['label']} - ({int(coords[0])}, {int(coords[1])}) {int(coords[2]-coords[0])}x{int(coords[3]-coords[1])}"
            
            # Add analysis info if available
            if roi['analysis_results']:
                if 'positive_area_percent' in roi['analysis_results']:
                    info += f" - Pos: {roi['analysis_results']['positive_area_percent']:.1f}%"
                    
            self.roi_listbox.insert(tk.END, info)
            
    def save_action_state(self):
        state = {
            'roi_count': len(self.roi_rects),
            'roi_data': [roi.copy() for roi in self.roi_rects]
        }
        
        if self.history_pointer < len(self.action_history) - 1:
            self.action_history = self.action_history[:self.history_pointer + 1]
            
        self.action_history.append(state)
        self.history_pointer += 1
        
        if len(self.action_history) > 20:
            self.action_history.pop(0)
            self.history_pointer -= 1
            
    def undo_action(self, event=None):
        if self.history_pointer > 0:
            self.history_pointer -= 1
            self.restore_state(self.action_history[self.history_pointer])
            
    def redo_action(self, event=None):
        if self.history_pointer < len(self.action_history) - 1:
            self.history_pointer += 1
            self.restore_state(self.action_history[self.history_pointer])
            
    def restore_state(self, state):
        for roi in self.roi_rects:
            self.canvas.delete(roi['rect_id'])
        
        self.roi_rects = []
        for roi_data in state['roi_data']:
            display_coords = self.original_to_display_coords(roi_data['coords'])
            rect_id = self.canvas.create_rectangle(
                *display_coords, outline='red', width=2, fill='', stipple='gray25'
            )
            
            new_roi = roi_data.copy()
            new_roi['rect_id'] = rect_id
            new_roi['display_coords'] = display_coords
            self.roi_rects.append(new_roi)
            
        self.deselect_all_rois()
        self.update_roi_list()
        self.update_status()
        
    def clear_action_history(self):
        self.action_history = []
        self.history_pointer = -1
        
    def update_status(self):
        roi_count = len(self.roi_rects)
        max_rois = self.num_rois.get()
        mode = self.interaction_mode.get()
        zoom_text = f"Zoom: {self.zoom_level:.1f}x" if self.zoom_level != 1.0 else ""
        
        if self.selected_roi:
            selected_text = f"Selected: {self.selected_roi['label']}"
        else:
            selected_text = ""
            
        status_parts = [f"ROIs: {roi_count}/{max_rois}", f"Mode: {mode.title()}"]
        if zoom_text:
            status_parts.append(zoom_text)
        if selected_text:
            status_parts.append(selected_text)
        if self.analysis_results:
            status_parts.append(f"Analyzed: {len(self.analysis_results)}")
            
        self.status_var.set(" | ".join(status_parts))


def main():
    """Main function to run the IHC ROI Analyzer"""
    # Check for required dependencies
    try:
        import cv2
        import numpy as np
        from skimage import color, segmentation, measure
        from scipy import ndimage
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install opencv-python numpy scikit-image scipy matplotlib")
        return
    
    root = tk.Tk()
    app = IHC_ROI_Analyzer(root)
    
    # Handle window resize
    def on_configure(event):
        if event.widget == app.canvas and app.image:
            app.root.after_idle(app.display_image)
    
    app.canvas.bind("<Configure>", on_configure)
    
    # Show welcome message
    welcome_msg = """
Welcome to IHC ROI Analysis Tool!

This tool combines ROI selection with advanced IHC analysis capabilities:

1. Load your IHC image (DAB/H&E stained)
2. Define regions of interest (ROIs) 
3. Run automated IHC analysis including:
   - Color deconvolution
   - Nuclei detection  
   - Staining quantification
   - H-score calculation

Press F5 to run analysis on all ROIs, or right-click individual ROIs for targeted analysis.

Key Features:
• Multiple stain types (DAB, H&E, Sirius Red)
• Automated positive area detection
• Nuclei counting and classification
• Statistical summaries and export options
• Save/load ROI templates for batch processing

Get started by opening an IHC image from the File menu!
    """
    
    def show_welcome():
        messagebox.showinfo("Welcome to IHC ROI Analysis Tool", welcome_msg)
    
    # Show welcome after a short delay
    root.after(1000, show_welcome)
    
    root.mainloop()


if __name__ == "__main__":
    main()