from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import backend_bases
import fitz  # PyMuPDF
import numpy as np
import cv2
import pickle
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from tkinter import filedialog
import subprocess, sys
import tkinter as tk
import threading
import pandas as pd

class ImageCropper:
    def __init__(self, images, bboxes):
        self.images = images
        self.bboxes = bboxes
        self.current_idx = 0
        self.n_images = len(images)
        self.cropped_images = []  # List to store cropped images
        self.fig, self.ax = plt.subplots()
        self.rect = None
        self.press = None
        self.current_bbox = self.bboxes[self.current_idx]
        self.rect_patch = None
        
        # Display first image with bbox
        self.update_image()

        # Connect the mouse event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Connect the navigation event from the toolbar
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        ImageCropper.plt_maximize()
        plt.show()

    def update_image(self):
        """Display the current image with a bounding box."""
        self.ax.clear()
        self.ax.imshow(self.images[self.current_idx], cmap='gray')

        self.ax.axis('off')
        self.ax.set_frame_on(True)
        self.ax.set_title(f"Image {self.current_idx + 1} / {self.n_images}")

        # Draw the bounding box
        if self.rect_patch:
            self.rect_patch.remove()

        if self.current_bbox:
            img_height, img_width = self.images[self.current_idx].shape[:2]
            blend = 0.3

            # Step 1: Add four dark patches to cover the area outside the bounding box
            # Top patch (above the bounding box)
            top_patch = Rectangle((0, 0), img_width, self.current_bbox[1], 
                                  linewidth=0, edgecolor=None, facecolor='black', alpha=blend)
            self.ax.add_patch(top_patch)

            # Bottom patch (below the bounding box)
            bottom_patch = Rectangle((0, self.current_bbox[1] + self.current_bbox[3]), img_width, img_height - (self.current_bbox[1] + self.current_bbox[3]), 
                                     linewidth=0, edgecolor=None, facecolor='black', alpha=blend)
            self.ax.add_patch(bottom_patch)

            # Left patch (left of the bounding box)
            left_patch = Rectangle((0, self.current_bbox[1]), self.current_bbox[0], self.current_bbox[3], 
                                   linewidth=0, edgecolor=None, facecolor='black', alpha=blend)
            self.ax.add_patch(left_patch)

            # Right patch (right of the bounding box)
            right_patch = Rectangle((self.current_bbox[0] + self.current_bbox[2], self.current_bbox[1]), img_width - (self.current_bbox[0] + self.current_bbox[2]), self.current_bbox[3], 
                                    linewidth=0, edgecolor=None, facecolor='black', alpha=blend)
            self.ax.add_patch(right_patch)

            # Step 2: Add the bounding box itself (red outline)
            rect_patch = Rectangle((self.current_bbox[0], self.current_bbox[1]), 
                                   self.current_bbox[2], self.current_bbox[3], 
                                   linewidth=0.1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect_patch)

        help_text = "Click and drag to crop. Use scroll or arrow keys to move between images. \nClose the window to save all images."
        self.ax.text(0.5, -0.05, help_text, transform=self.ax.transAxes, fontsize=9, 
                 verticalalignment='top', horizontalalignment='center')
        plt.draw()

    def on_press(self, event):
        """Record the initial position when the mouse button is pressed."""
        if event.inaxes != self.ax:
            return
        self.press = (event.xdata, event.ydata)

    def on_motion(self, event):
        """Update the size of the bounding box during dragging, constrained within image bounds."""
        if self.press is None:
            return

        # Get image dimensions
        img_height, img_width = self.images[self.current_idx].shape[:2]

        # Check if the mouse has left the axes (i.e., outside the image area)
        if event.xdata is None or event.ydata is None:
            # Simulate a release when mouse goes outside the image
            self.finalize_bbox_on_exit()
            return

        x0, y0 = self.press
        x1, y1 = event.xdata, event.ydata

        # Constrain the coordinates within the image dimensions
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))

        # Update the bounding box based on the constrained drag
        self.current_bbox = [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)]
        self.update_image()

    def finalize_bbox_on_exit(self):
        """Simulate a release event if the mouse goes outside the image area."""
        if self.press is None:
            return

        # Get image dimensions
        img_height, img_width = self.images[self.current_idx].shape[:2]

        # Finalize the bounding box coordinates by constraining to the image bounds
        x1, y1 = self.current_bbox[2], self.current_bbox[3]  # Get the current bbox dimensions
        self.current_bbox = [max(0, min(self.current_bbox[0], img_width - 1)),
                             max(0, min(self.current_bbox[1], img_height - 1)),
                             x1, y1]
        self.bboxes[self.current_idx] = self.current_bbox

        # Reset press state
        self.press = None
        self.update_image()

    def on_release(self, event):
        """Finalize the bounding box when the mouse button is released, even if outside the image area."""
        if self.press is None:
            return

        # Get image dimensions
        img_height, img_width = self.images[self.current_idx].shape[:2]


        # Finalize the bounding box coordinates
        x0, y0 = self.press
        # Constrain release coordinates to be within the image bounds
        x1 = max(0, min(event.xdata, img_width - 1)) if event.xdata else self.current_bbox[0] + self.current_bbox[2]
        y1 = max(0, min(event.ydata, img_height - 1)) if event.ydata else self.current_bbox[1] + self.current_bbox[3]

        self.current_bbox = [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)]
        self.bboxes[self.current_idx] = self.current_bbox

        # Reset press state
        self.press = None
        self.update_image()

    def save_cropped_image(self):
        """Crop the current image based on the bounding box and store it."""
        x, y, w, h = map(int, self.current_bbox)
        cropped_img = self.images[self.current_idx][y:y+h, x:x+w]
        self.cropped_images.append(cropped_img)

    def on_key_press(self, event):
        """Move between images using keyboard arrows."""
        if event.key == 'right':  # Next image
            self.next_image()
        elif event.key == 'left':  # Previous image
            self.previous_image()

    def on_scroll(self, event):
        """Move between images using scroll wheel."""
        if event.button == 'up':
            self.previous_image()
        elif event.button == 'down':
            self.next_image()

    def next_image(self):
        """Move to the next image in the list."""
        if self.current_idx < self.n_images - 1:
            self.current_idx += 1
            self.current_bbox = self.bboxes[self.current_idx]
            self.update_image()

    def previous_image(self):
        """Move to the previous image in the list."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.current_bbox = self.bboxes[self.current_idx]
            self.update_image()

    def get_cropped_images(self):
        """Return the list of cropped images."""
        cropped_images = []
        for image, bbox in zip(self.images, self.bboxes):
            x, y, w, h = map(int, bbox)
            cropped_img = image[y:y+h, x:x+w]
            cropped_images.append(cropped_img)
        
        return cropped_images

    @staticmethod
    def plt_maximize():
        manager = plt.get_current_fig_manager()

        # For TkAgg backend (cross-platform)
        if sys.platform.startswith('win') or sys.platform.startswith('linux') or sys.platform == 'darwin':
            try:
                screen_width = manager.window.winfo_screenwidth()
                screen_height = manager.window.winfo_screenheight()
    
                # Set window size to 90% of the screen size
                manager.window.geometry(f"{int(screen_width * 0.9)}x{int(screen_height * 0.9)}+0+0")
            except AttributeError:
                print("Could not maximize the window. Ensure you're using a compatible backend.")
        else:
            raise RuntimeError("plt_maximize() is not implemented for current backend:", backend)

class FilePicker:
    @staticmethod
    def pick_pdf_file():
        """Pick a PDF file based on the platform."""
        platform = sys.platform
        if platform.startswith('linux'):
            if FilePicker._is_zenity_available():
                return FilePicker._pick_file_gnome("*.pdf", "PDF files")
            else:
                return FilePicker._pick_file_tkinter([("PDF files", "*.pdf"), ("All files", "*.*")])

        elif platform.startswith('win'):
            return FilePicker._pick_file_windows("PDF files\0*.pdf\0All files\0*.*\0")

        elif platform == 'darwin':
            return FilePicker._pick_file_tkinter([("PDF files", "*.pdf"), ("All files", "*.*")])
        else:
            print("Unsupported platform")
            return None

    @staticmethod
    def pick_xlsx_file():
        """Pick an Excel (.xlsx) file based on the platform."""
        platform = sys.platform
        if platform.startswith('linux'):
            if FilePicker._is_zenity_available():
                return FilePicker._pick_file_gnome("*.xlsx", "Excel files")
            else:
                return FilePicker._pick_file_tkinter([("Excel files", "*.xlsx"), ("All files", "*.*")])

        elif platform.startswith('win'):
            return FilePicker._pick_file_windows("Excel files\0*.xlsx\0All files\0*.*\0")

        elif platform == 'darwin':
            return FilePicker._pick_file_tkinter([("Excel files", "*.xlsx"), ("All files", "*.*")])
        else:
            print("Unsupported platform")
            return None

    @staticmethod
    def pick_directory():
        """Pick a directory based on the platform."""
        platform = sys.platform
        if platform.startswith('linux'):
            if FilePicker._is_zenity_available():
                return FilePicker._pick_directory_gnome()
            else:
                return FilePicker._pick_directory_tkinter()

        elif platform.startswith('win'):
            return FilePicker._pick_directory_windows()

        elif platform == 'darwin':
            return FilePicker._pick_directory_tkinter()
        else:
            print("Unsupported platform")
            return None

    @staticmethod
    def _is_zenity_available():
        """Check if zenity is available on the system (Linux GNOME)."""
        return subprocess.call(['which', 'zenity'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

    @staticmethod
    def _pick_file_gnome(file_extension, file_type_name):
        """Use zenity to pick a file in GNOME based on the provided file extension."""
        command = [
            'zenity',
            '--file-selection',
            '--title=Select a file',
            f'--file-filter={file_extension}',
            '--file-filter=All files | *.*'
        ]
        try:
            file_path = subprocess.check_output(command, stderr=subprocess.DEVNULL).strip().decode('utf-8')
            return file_path
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def _pick_file_windows(filter_string):
        """Use native Windows file picker to pick a file based on the provided filter."""
        import ctypes
        from ctypes import wintypes
        
        OFN_FILEMUSTEXIST = 0x1000
        buffer = ctypes.create_unicode_buffer(260)

        open_file_name = ctypes.windll.comdlg32.GetOpenFileNameW
        open_file_name.restype = wintypes.BOOL

        class OPENFILENAME(ctypes.Structure):
            _fields_ = [
                ("lStructSize", wintypes.DWORD),
                ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HINSTANCE),
                ("lpstrFilter", wintypes.LPCWSTR),
                ("lpstrCustomFilter", wintypes.LPWSTR),
                ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPWSTR),
                ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPWSTR),
                ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCWSTR),
                ("lpstrTitle", wintypes.LPCWSTR),
                ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD),
                ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR),
                ("lCustData", wintypes.LPARAM),
                ("lpfnHook", wintypes.LPVOID),
                ("lpTemplateName", wintypes.LPCWSTR),
            ]
        
        dialog_struct = OPENFILENAME()
        dialog_struct.lStructSize = ctypes.sizeof(OPENFILENAME)
        dialog_struct.lpstrFilter = filter_string
        dialog_struct.lpstrFile = buffer
        dialog_struct.nMaxFile = 260
        dialog_struct.Flags = OFN_FILEMUSTEXIST

        if open_file_name(ctypes.byref(dialog_struct)):
            return buffer.value
        else:
            return None

    @staticmethod
    def _pick_file_tkinter(filetypes):
        """Use tkinter as a fallback to pick a file."""
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=filetypes
        )

        if file_path:
            print(f"File selected: {file_path}")
            return file_path
        else:
            print("No file selected")
            return None

    @staticmethod
    def _pick_directory_gnome():
        """Use zenity to pick a directory in GNOME."""
        command = [
            'zenity', 
            '--file-selection', 
            '--directory', 
            '--title=Select a directory'
        ]
        try:
            folder_path = subprocess.check_output(command, stderr=subprocess.DEVNULL).strip().decode('utf-8')
            return folder_path
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def _pick_directory_windows():
        """Use tkinter or ctypes for picking directories in Windows."""
        import ctypes
        from ctypes import wintypes
        
        BIF_RETURNONLYFSDIRS = 0x0001
        BIF_NEWDIALOGSTYLE = 0x0040
        MAX_PATH = 260

        buffer = ctypes.create_unicode_buffer(MAX_PATH)

        class BROWSEINFO(ctypes.Structure):
            _fields_ = [
                ("hwndOwner", wintypes.HWND),
                ("pidlRoot", wintypes.LPCVOID),
                ("pszDisplayName", wintypes.LPWSTR),
                ("lpszTitle", wintypes.LPCWSTR),
                ("ulFlags", wintypes.UINT),
                ("lpfn", wintypes.LPVOID),
                ("lParam", wintypes.LPARAM),
                ("iImage", wintypes.INT),
            ]

        SHBrowseForFolderW = ctypes.windll.shell32.SHBrowseForFolderW
        SHBrowseForFolderW.restype = wintypes.LPCVOID

        SHGetPathFromIDListW = ctypes.windll.shell32.SHGetPathFromIDListW
        SHGetPathFromIDListW.argtypes = [wintypes.LPCVOID, wintypes.LPWSTR]
        SHGetPathFromIDListW.restype = wintypes.BOOL

        browse_info = BROWSEINFO()
        browse_info.pszDisplayName = buffer
        browse_info.lpszTitle = "Select a folder"
        browse_info.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE

        pidl = SHBrowseForFolderW(ctypes.byref(browse_info))
        if pidl and SHGetPathFromIDListW(pidl, buffer):
            return buffer.value
        else:
            return None

    @staticmethod
    def _pick_directory_tkinter():
        """Use tkinter as a fallback to pick a directory."""
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        folder_path = filedialog.askdirectory(title="Select a folder")

        if folder_path:
            print(f"Folder selected: {folder_path}")
            return folder_path
        else:
            print("No folder selected")
            return None

class PDFViewerApp(tk.Tk):
    def __init__(self, pdf_images=[]):
        super().__init__()
        self.image_list, self.bbox_list = None, None
        self.title("PDF Viewer")
        self.geometry("800x600")
        self.pdf_images = pdf_images

        # GNOME Light Theme Colors
        self.bg_color = "#FFFFFF"   # Light gray background
        self.btn_color = "#3584E4"  # Light blue for buttons
        self.textbox_bg_color = "#FFFFFF"  # White for the textbox
        self.scrollbar_color = "#D3D3D3"   # Light gray for the scrollbar
        self.btn_hover_color = "#4A90E2"  # Slightly darker blue for hover
        self.border_color = "#E5E5E5"  # Light gray for borders
        self.active_bg_color = "#3584E4"  # Active button blue

        self.configure(bg=self.bg_color)

        # Create the UI components
        self.create_widgets()

        # Flags and Threads
        self.stop_loading_flag = False  # Renamed variable to avoid conflict
        self.loading_thread = None

        self.filtered_images = None

    def create_widgets(self):
        """Create and layout the widgets for the PDF viewer."""
        # Top frame containing textbox and load button (centered horizontally)
        top_frame = tk.Frame(self, bg=self.bg_color)
        top_frame.pack(pady=10)

        # Centering using grid
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(2, weight=1)

        # Textbox for displaying selected PDF path
        self.textbox = tk.Text(top_frame, height=1, width=65, state='disabled', bg=self.textbox_bg_color, font=('Arial', 12))
        self.textbox.grid(row=0, column=1, padx=10)

        # Load PDF button with 📄 Unicode icon (centered)
        self.load_button_text = tk.StringVar()
        self.load_button = tk.Button(top_frame, textvariable=self.load_button_text, command=self.load_pdf, bg=self.btn_color, fg='white', 
                                     font=('Arial', 12), bd=0, activebackground=self.btn_hover_color, width=12)
        self.load_button_text.set("Select 📄")
        self.load_button.grid(row=0, column=2)

        # Dropdowns for start, stop, and step
        control_frame = tk.Frame(self, bg=self.bg_color)
        control_frame.pack(pady=5)

        start_label = tk.Label(control_frame, text="Start Page:", bg=self.bg_color)
        start_label.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        self.start_page = ttk.Combobox(control_frame, state="disabled", width=10, values=[])
        self.start_page.pack(side=tk.LEFT, padx=(10, 5), pady=3)
        stop_label = tk.Label(control_frame, text="Stop Page:", bg=self.bg_color)
        stop_label.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        self.stop_page = ttk.Combobox(control_frame, state="disabled", width=10, values=[])
        self.stop_page.pack(side=tk.LEFT, padx=(10, 5), pady=3)
        step_label = tk.Label(control_frame, text="Step Pages:", bg=self.bg_color)
        step_label.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        self.step_pages = ttk.Combobox(control_frame, state="disabled", width=10, values=[])
        self.step_pages.pack(side=tk.LEFT, padx=(10, 5), pady=3)

        # Style the Combobox with padding and a modern appearance
        style = ttk.Style()
        style.configure("TCombobox", padding=5, relief="flat", arrowcolor=self.bg_color, background=self.active_bg_color, lightcolor=self.bg_color)

        self.start_page.pack(side=tk.LEFT, padx=10, pady=5)
        self.stop_page.pack(side=tk.LEFT, padx=10, pady=5)
        self.step_pages.pack(side=tk.LEFT, padx=10, pady=5)

        self.start_page.bind("<<ComboboxSelected>>", self.update_thumbnails)
        self.stop_page.bind("<<ComboboxSelected>>", self.update_thumbnails)
        self.step_pages.bind("<<ComboboxSelected>>", self.update_thumbnails)

        # Frame for thumbnail display with a vertical scroll bar
        self.thumbnail_frame = tk.Frame(self, bg=self.bg_color)
        self.thumbnail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for scrolling
        self.canvas = tk.Canvas(self.thumbnail_frame, bg=self.bg_color)
        self.scrollbar = ttk.Scrollbar(self.thumbnail_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Scroll frame for thumbnails inside the canvas
        self.scroll_frame = tk.Frame(self.canvas, bg=self.bg_color)
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')

        # Bind the frame's configure event to update the scroll region
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Bind mouse wheel scrolling
        self.bind_mouse_wheel()

        # Pack the canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Process button below thumbnails
        self.process_button = tk.Button(self, text="Process", command=self.process_and_return_images, 
                                        bg=self.btn_color, fg="white", font=('Arial', 12), state=tk.DISABLED,
                                        bd=0, activebackground=self.btn_hover_color, padx=20, pady=10, width=20)
        self.process_button.pack(pady=10)

        # Bind resize to update thumbnails layout
        self.canvas.bind("<Configure>", lambda e: self.update_thumbnails())
    
    def bind_mouse_wheel(self):
        """Bind mouse wheel scrolling for Windows and Linux."""
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind_all("<Button-4>", lambda event: self.on_mouse_wheel(event, -1))  # Linux (scroll up)
        self.canvas.bind_all("<Button-5>", lambda event: self.on_mouse_wheel(event, 1))  # Linux (scroll down)

    def on_mouse_wheel(self, event, direction=None):
        """Scroll the canvas based on mouse wheel movement."""
        if direction is None:  # For Windows
            direction = -1 if event.delta > 0 else 1
        
        # Scroll the canvas
        self.canvas.yview_scroll(int(direction), "units")

    def get_pdf_as_images(self):
        """
        Extracts each page of a PDF as a NumPy array

        Parameters:
        - pdf_path (str): Path to the PDF file.

        Returns:
        - images (list): A list of NumPy arrays, each representing a page image.
        """
        # Open the PDF
        pdf_document = fitz.open(self.pdf_path)
        num_pages = pdf_document.page_count
        images = []

        for page_num in range(num_pages):
            self.load_button_text.set(f"Loading [{round(page_num * 100 / num_pages)}%]")
            self.update_idletasks()

            # Get the page
            page = pdf_document.load_page(page_num)

            # Extract the page as a pixmap (no DPI scaling needed)
            pix = page.get_pixmap()

            # Create a PIL image from the pixmap
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert the PIL image to a NumPy array
            img_np = np.array(img_pil)

            # Append the NumPy array to the images list
            images.append(img_np)

        # Close the document
        pdf_document.close()

        return images

    def load_pdf(self):
        """Load a PDF and display its thumbnails."""
        self.load_button.config(state=tk.DISABLED)
        self.load_button_text.set("Loading ...")
        self.update_idletasks()
        self.pdf_path = FilePicker.pick_pdf_file()

        if self.pdf_path:
            self.textbox.config(state=tk.NORMAL)
            self.textbox.delete("1.0", tk.END)
            self.textbox.insert(tk.END, self.pdf_path)
            self.textbox.config(state=tk.DISABLED)
            

            # Simulate loading PDF as images
            self.pdf_images = self.get_pdf_as_images()

            # Update the dropdowns for start, stop, and step
            num_pages = len(self.pdf_images)
            self.start_page["values"] = list(range(1, num_pages + 1))
            self.stop_page["values"] = list(range(1, num_pages + 1))[::-1]
            self.step_pages["values"] = list(range(1, num_pages + 1))

            self.start_page.current(0)
            self.stop_page.current(0)
            self.step_pages.current(0)
            
            self.disable_filters()

            # Start lazy loading thumbnails
            self.lazy_load_thumbnails(0, num_pages, 1)
        else:
            self.pdf_images = []
            self.disable_filters()
        
        self.load_button_text.set("Select 📄")
        self.load_button.config(state=tk.NORMAL)
    
    def enable_filters(self):
        self.start_page['state'] = 'readonly'
        self.stop_page['state'] = 'readonly'
        self.step_pages['state'] = 'readonly'
        self.process_button.config(state=tk.NORMAL)


    def disable_filters(self):
        self.start_page['state'] = "disabled"
        self.stop_page['state'] = "disabled"
        self.step_pages['state'] = "disabled"
        self.process_button.config(state=tk.DISABLED)

    def lazy_load_thumbnails(self, start, stop, step):
        """Load thumbnails progressively in a separate thread."""
        # Clear current thumbnails before loading new ones
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        # Show loading indicator
        self.loading_label = tk.Label(self.scroll_frame, text="Loading...", bg=self.bg_color, font=('Arial', 14))
        self.loading_label.pack(pady=20)

        # Ensure old loading process is stopped
        self.stop_loading()

        # Reset stop_loading_flag
        self.stop_loading_flag = False

        # Start new loading thread
        def load():
            thumbnails_per_row = max(1, self.canvas.winfo_width() // (150 + 10))  # Fit as many as possible in a row
            total_height = 0
            row_frame = None

            counter = 0
            for i in range(start, stop, step):
                img_np = self.pdf_images[i]
                if self.stop_loading_flag:
                    break

                # Create thumbnail and add to UI
                img_pil = Image.fromarray(img_np)
                img_pil.thumbnail((150, 200))
                img_tk = ImageTk.PhotoImage(img_pil)

                if counter % thumbnails_per_row == 0:
                    row_frame = tk.Frame(self.scroll_frame, bg=self.bg_color)
                    row_frame.pack(fill=tk.X)  # Pack the row frame horizontally
                    total_height += 210

                # Create a frame for each image and its label (image name)
                image_frame = tk.Frame(row_frame, bg=self.bg_color)
                image_frame.pack(side=tk.LEFT, padx=5, pady=5)  # Pack each image and label frame

                # Add the image thumbnail
                label_img = tk.Label(image_frame, image=img_tk)
                label_img.image = img_tk  # Keep a reference to prevent garbage collection
                label_img.pack()

                # Add the image name under the thumbnail
                label_name = tk.Label(image_frame, text=str(i + 1), bg=self.bg_color, anchor='center')
                label_name.pack()


                # Ensure updates happen progressively
                self.scroll_frame.update_idletasks()
                counter += 1
            

            if self.stop_loading_flag:
                return

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            # Set the scrollbar to be visible if needed
            if total_height > self.canvas.winfo_height():
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Show the scrollbar
            else:
                self.scrollbar.pack_forget()

            # Hide loading indicator when done
            self.loading_label.pack_forget()
            self.enable_filters()

        self.loading_thread = threading.Thread(target=load)
        self.loading_thread.start()

    def stop_loading(self):
        """Stop any ongoing thumbnail loading process."""
        while hasattr(self, "loading_thread") and self.loading_thread and self.loading_thread.is_alive():
            self.stop_loading_flag = True
            
    def process_and_return_images(self):
        """Process the PDF, filter the images, and open the ImageCropper with a new Tkinter window."""
        self.disable_filters()
        self.process_button.config(text="")
        start = int(self.start_page.get()) - 1
        stop = int(self.stop_page.get())
        step = int(self.step_pages.get())
        self.filtered_images = self.pdf_images[start:stop:step]

        if not self.filtered_images:
            return
        
        self.image_list, self.bbox_list = [], []
        for i, image in enumerate(self.filtered_images):
            image, bbox = get_crop_estimate(image)
            self.image_list.append(image)
            self.bbox_list.append(bbox)
            self.process_button.config(text=f"Processing [{round(i*100/len(self.filtered_images))}%]")
            self.update_idletasks()

        self.destroy()

    def update_thumbnails(self, *args):
        """Update thumbnails when start, stop, or step is changed."""
        # Stop any previous loading process
        self.stop_loading()

        self.disable_filters()
        # Get start, stop, and step values from dropdowns
        if len(self.pdf_images) != 0:
            start = int(self.start_page.get()) - 1
            stop = int(self.stop_page.get())
            step = int(self.step_pages.get())


            # Start lazy loading with the new filter
            self.lazy_load_thumbnails(start, stop, step)


class ShowCroppedImages(tk.Tk):
    def __init__(self, cropped_images, selected_folder=None, selected_excel=None):
        super().__init__()
        self.image_list = cropped_images
        self.selected_folder = selected_folder
        self.selected_excel = selected_excel
        self.image_names = []
        self.get_image_names()
        self.title("Cropped Images Viewer")
        self.geometry("800x600")

        # GNOME Light Theme Colors
        self.bg_color = "#FFFFFF"
        self.btn_color = "#3584E4"
        self.scrollbar_color = "#D3D3D3"
        self.btn_hover_color = "#4A90E2"
        self.border_color = "#E5E5E5"
        self.active_bg_color = "#3584E4"

        self.configure(bg=self.bg_color)

        background = Image.new('RGB', (150, 200), (0, 0, 0))  # Black background

        # Create a drawing context for the background
        draw = ImageDraw.Draw(background)

        # Draw white diagonal stripes on the black background
        for i in range(-200, 150, 4):  # Loop for drawing stripes
            draw.line((i, 0, i + 200, 200), fill=(255, 255, 255), width=1)

        self.background = background

        # Create UI components
        self.create_widgets()

        # Flags and Threads
        self.stop_loading_flag = False
        self.loading_thread = None
        

    def create_widgets(self):
        """Create and layout widgets for displaying cropped images."""

        # Frame for thumbnail display with a vertical scroll bar
        self.thumbnail_frame = tk.Frame(self, bg=self.bg_color)
        self.thumbnail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for scrolling
        self.canvas = tk.Canvas(self.thumbnail_frame, bg=self.bg_color)
        self.scrollbar = ttk.Scrollbar(self.thumbnail_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Scroll frame for thumbnails inside the canvas
        self.scroll_frame = tk.Frame(self.canvas, bg=self.bg_color)
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')

        # Bind the frame's configure event to update the scroll region
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Bind mouse wheel scrolling
        self.bind_mouse_wheel()

        # Pack the canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind resize to update thumbnails layout
        self.canvas.bind("<Configure>", lambda e: self.update_thumbnails())


        # Frame for folder selection at the bottom
        middle_frame = tk.Frame(self, bg=self.bg_color)
        middle_frame.pack(fill=tk.X, pady=10, padx=10)

        middle_frame.grid_columnconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(2, weight=1)

        # Text box to display the selected excel path
        self.excel_path_textbox = tk.Text(middle_frame, height=1, width=65, state='disabled', bg='#F0F0F0', font=('Arial', 12))
        self.excel_path_textbox.grid(row=0, column=0, padx=10)

        # Button to select excel
        self.select_excel_button = tk.Button(middle_frame, text="Select 📊", command=self.select_excel,
                                              bg=self.btn_color, fg='white', font=('Arial', 12), bd=0,
                                              activebackground=self.btn_hover_color)
        self.select_excel_button.grid(row=0, column=1, padx=10)

        # Frame for folder selection at the bottom
        bottom_frame = tk.Frame(self, bg=self.bg_color)
        bottom_frame.pack(fill=tk.X, pady=10, padx=10)

        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)

        # Text box to display the selected folder path
        self.folder_path_textbox = tk.Text(bottom_frame, height=1, width=65, state='disabled', bg='#F0F0F0', font=('Arial', 12))
        self.folder_path_textbox.grid(row=0, column=0, padx=10)

        # Button to select folder
        self.select_folder_button = tk.Button(bottom_frame, text="Browse 📁", command=self.select_folder,
                                              bg=self.btn_color, fg='white', font=('Arial', 12), bd=0,
                                              activebackground=self.btn_hover_color)
        self.select_folder_button.grid(row=0, column=1, padx=10)

        # Save button to save images to the selected folder and exit
        self.save_button = tk.Button(self, text="Save", command=self.save_and_exit, width=20,
                                     bg=self.btn_color, fg='white', font=('Arial', 12), bd=0,
                                     activebackground=self.btn_hover_color, state=tk.DISABLED)
        self.save_button.pack(pady=10)
        self.check_selection()
    
    def get_image_names(self):
        self.image_names = []
        
        if self.selected_excel:
            df = pd.read_excel(self.selected_excel)
            for i, _ in enumerate(self.image_list):
                self.image_names.append(f"{df["subject"][i]}_{df["trial"][i]}")
        else:
            for i, _ in enumerate(self.image_list):
                self.image_names.append(f"rocf_{i+1}")

    def bind_mouse_wheel(self):
        """Bind mouse wheel scrolling for Windows and Linux."""
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind_all("<Button-4>", lambda event: self.on_mouse_wheel(event, -1))  # Linux (scroll up)
        self.canvas.bind_all("<Button-5>", lambda event: self.on_mouse_wheel(event, 1))  # Linux (scroll down)

    def on_mouse_wheel(self, event, direction=None):
        """Scroll the canvas based on mouse wheel movement."""
        if direction is None:  # For Windows
            direction = -1 if event.delta > 0 else 1

        # Scroll the canvas
        self.canvas.yview_scroll(int(direction), "units")

    def select_folder(self):
        """Open folder selection dialog and display the selected folder path in the textbox."""
        self.selected_folder = FilePicker.pick_directory()
        self.check_selection()

    def select_excel(self):
        """Open excel selection dialog and display the selected excel path in the textbox."""
        self.selected_excel = FilePicker.pick_xlsx_file()
        self.check_selection()
        self.get_image_names()
        self.update_thumbnails()

    def check_selection(self):
        if self.selected_folder:
            self.folder_path_textbox.config(state=tk.NORMAL)
            self.folder_path_textbox.delete("1.0", tk.END)
            self.folder_path_textbox.insert(tk.END, self.selected_folder)
            self.folder_path_textbox.config(state=tk.DISABLED)

            # Enable the Save button after a folder is selected
            self.save_button.config(state=tk.NORMAL)
        else:
            self.folder_path_textbox.config(state=tk.NORMAL)
            self.folder_path_textbox.delete("1.0", tk.END)
            self.folder_path_textbox.config(state=tk.DISABLED)

            # Enable the Save button after a folder is selected
            self.save_button.config(state=tk.NORMAL)
        
        if self.selected_excel:
            self.excel_path_textbox.config(state=tk.NORMAL)
            self.excel_path_textbox.delete("1.0", tk.END)
            self.excel_path_textbox.insert(tk.END, self.selected_excel)
            self.excel_path_textbox.config(state=tk.DISABLED)
        else:
            self.excel_path_textbox.config(state=tk.NORMAL)
            self.excel_path_textbox.delete("1.0", tk.END)
            self.excel_path_textbox.config(state=tk.DISABLED)

    def lazy_load_thumbnails(self):
        """Load thumbnails progressively in a separate thread."""
        # Clear current thumbnails before loading new ones
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        # Show loading indicator
        self.loading_label = tk.Label(self.scroll_frame, text="Loading...", bg=self.bg_color, font=('Arial', 14))
        self.loading_label.pack(pady=20)

        # Ensure old loading process is stopped
        self.stop_loading()

        # Reset stop_loading_flag
        self.stop_loading_flag = False

        # Start new loading thread
        def load():
            thumbnails_per_row = max(1, self.canvas.winfo_width() // (150 + 10))  # Fit as many as possible in a row
            total_height = 0
            row_frame = None

            def create_thumbnail_with_padding(img_np):
                img_pil = Image.fromarray(img_np)  # Convert the NumPy array to a PIL image

                # Create the thumbnail (preserves aspect ratio but fits within 150x200)
                img_pil.thumbnail((150, 200))

                # Create a new image (canvas) with the desired size (150x200) and white background
                background = self.background.copy()

                # Calculate position to center the image
                offset_x = (150 - img_pil.width) // 2
                offset_y = (200 - img_pil.height) // 2

                # Paste the thumbnail onto the white canvas
                background.paste(img_pil, (offset_x, offset_y))

                return background

            for i, img_np in enumerate(self.image_list):
                if self.stop_loading_flag:
                    break
                
                # Create thumbnail and add to UI
                img_pil = create_thumbnail_with_padding(img_np)
                img_tk = ImageTk.PhotoImage(img_pil)

                # Create a new row frame if it's the start of a new row
                if i % thumbnails_per_row == 0:
                    row_frame = tk.Frame(self.scroll_frame, bg=self.bg_color)
                    row_frame.pack(fill=tk.X)  # Pack the row frame horizontally
                    total_height += 210  # Image + Label height

                # Create a frame for each image and its label (image name)
                image_frame = tk.Frame(row_frame, bg=self.bg_color)
                image_frame.pack(side=tk.LEFT, padx=5, pady=5)  # Pack each image and label frame

                # Add the image thumbnail
                label_img = tk.Label(image_frame, image=img_tk)
                label_img.image = img_tk  # Keep a reference to prevent garbage collection
                label_img.pack()

                # Add the image name under the thumbnail
                image_name = self.image_names[i] if i < len(self.image_names) else "No Name"  # Get image name, fallback if missing
                label_name = tk.Label(image_frame, text=image_name, bg=self.bg_color, anchor='center')
                label_name.pack()

                # Ensure updates happen progressively
                self.scroll_frame.update_idletasks()

            if self.stop_loading_flag:
                return

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            # Set the scrollbar to be visible if needed
            if total_height > self.canvas.winfo_height():
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Show the scrollbar
            else:
                self.scrollbar.pack_forget()

            # Hide loading indicator when done
            self.loading_label.pack_forget()


        self.loading_thread = threading.Thread(target=load)
        self.loading_thread.start()

    def stop_loading(self):
        """Stop any ongoing thumbnail loading process."""
        while hasattr(self, "loading_thread") and self.loading_thread and self.loading_thread.is_alive():
            self.stop_loading_flag = True

    def update_thumbnails(self):
        """Update thumbnails when window size is changed."""
        # Stop any previous loading process
        self.stop_loading()

        # Start lazy loading with the existing images
        self.lazy_load_thumbnails()

    def save_and_exit(self):
        """Save the images to the selected folder and exit the app."""
        if self.selected_folder and self.image_list:
            save_path = Path(self.selected_folder)
            save_path.mkdir(exist_ok=True)
            self.save_button.config(state="disabled")
            for i, img_np in enumerate(self.image_list):
                img_pil = Image.fromarray(img_np)
                img_pil.save(save_path / f"{self.image_names[i]}.png")
                self.save_button.config(text=f"Saving [{round(i * 100/ len(self.image_list))}%]")
                self.update_idletasks()

        # Close the app after saving
        self.destroy()




def process_image(image):
    smooth = cv2.medianBlur(image, 101)
    smooth = cv2.GaussianBlur(smooth, (31,31), sigmaX=1.0, sigmaY=1.0)
    std_img = cv2.divide(image, smooth, scale=255)

    lab = cv2.cvtColor(std_img, cv2.COLOR_BGR2Lab)

    # Split LAB channels to get the Luminance channel (L)
    L, A, B = cv2.split(lab)
    img_gray = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 7)
    kernel = np.ones((2,2))
    opened = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)

def get_crop_estimate(image):
    pr_img = process_image(image)
    height, width = image.shape[:2]
    binary = cv2.cvtColor(pr_img, cv2.COLOR_RGB2GRAY)
    binary = cv2.bitwise_not(binary)
    num_labels, labels_im = cv2.connectedComponents(binary, connectivity=8)

    # Find the largest connected component (excluding the background)
    largest_component = sorted(range(0, num_labels), key=lambda x: np.sum(labels_im == x))[-2]

    # Create a mask for the largest component
    mask = (labels_im == largest_component).astype(np.uint8) * 255
    # Get the bounding box for the largest component
    coords = np.column_stack(np.nonzero(mask))
    x, y, w, h = cv2.boundingRect(coords)

    padding_x = int(0.05 * height)
    padding_y = int(0.05 * width)
    
    # Calculate new bounding box with padding
    x_padded = max(0, x - padding_x)  # Ensure we don't go below 0
    y_padded = max(0, y - padding_y)
    w_padded = min(height - x_padded, w + 2 * padding_x)  # Clip to image width
    h_padded = min(width - y_padded, h + 2 * padding_y)
    
    return image, [y_padded, x_padded, h_padded, w_padded]


# Example usage
if __name__ == "__main__":
    backup_file = Path("restore.pkl")

    if not backup_file.exists():
        app = PDFViewerApp()
        app.mainloop()

        if not (app.image_list and app.bbox_list):
            exit(0)

        cropper = ImageCropper(app.image_list, app.bbox_list)
        cropped_images = cropper.get_cropped_images()
        pic_dir = Path(app.pdf_path).with_suffix("")
    else:
        print(f"Backup file '{backup_file}' exists. Using it to continue!")
        with open(backup_file, "rb") as f:
            cropped_images, pic_dir = pickle.load(f)
        backup_file.unlink()

    try:
        app = ShowCroppedImages(cropped_images, str(pic_dir.absolute()))
        app.mainloop()
    except Exception as e:
        with open(backup_file, "wb") as f:
            pickle.dump((cropped_images, pic_dir), f)
        raise e
