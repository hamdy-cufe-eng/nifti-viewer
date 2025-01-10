import sys
import numpy as np
import itk
import vtk
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QSlider, QFrame, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QPushButton, QToolBar, QStatusBar, QMessageBox, QTabWidget, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QIcon
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class ImageLoaderThread(QThread):
    """
    Thread for loading NIfTI images to prevent GUI freezing.
    """
    image_loaded = pyqtSignal(np.ndarray, object)  # Emits the image array and ITK image object
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            image = itk.imread(self.file_path)
            image_array = itk.GetArrayFromImage(image)
            self.image_loaded.emit(image_array, image)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ImageViewer(QWidget):
    """
    2D Orthogonal Image Viewer with Crosshairs .
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.brightness = {"xy": 0, "xz": 0, "zy": 0}  # Brightness adjustments
        self.contrast = {"xy": 1, "xz": 1, "zy": 1}  # Contrast multipliers
        self.initUI()
        self.image_array = None
        self.current_crosshair = None
        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_crosshair)
        self.timer.start()
        self.label = QLabel()
        self.zoom_level_xy = 0.4
        self.zoom_level_xz = 0.4
        self.zoom_level_yz = 0.4
        self.start_pos = None
        self.initial_brightness = 1.0
        self.initial_contrast = 1.0
        self.playing_state = {"xy": False, "xz": False, "yz": False}
        # Create timers for each view
        self.timers = {"xy": QTimer(self), "xz": QTimer(self), "yz": QTimer(self)}
        self.timers["yz"].timeout.connect(lambda: self.update_slice("yz"))
        # Connect timers to slice updating function
        self.timers["xy"].timeout.connect(lambda: self.update_slice("xy"))
        self.timers["xz"].timeout.connect(lambda: self.update_slice("xz"))

    def create_play_icon(self):
        # Create a green play icon
        pixmap = QPixmap(30, 30)
        pixmap.fill(Qt.transparent)  # Fill with transparent background
        painter = QPainter(pixmap)
        painter.setBrush(QColor(9, 132, 227))  # Set brush color to rgb(9, 132, 227)
        painter.setPen(Qt.transparent)  # No outline
        # Draw a play triangle
        triangle = [QPoint(5, 5), QPoint(25, 15), QPoint(5, 25)]
        painter.drawConvexPolygon(*triangle)  # Use drawConvexPolygon to draw the triangle
        painter.end()
        return QIcon(pixmap)  # Return as QIcon

    def create_pause_icon(self):
        # Create a pause icon (two vertical bars)
        pixmap = QPixmap(30, 30)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 0, 0))  # Set brush color to green
        painter.setPen(Qt.transparent)  # No outline
        # Draw two vertical bars for pause
        painter.drawRect(10, 5, 5, 20)
        painter.drawRect(20, 5, 5, 20)
        painter.end()
        return QIcon(pixmap)  # Return as QIcon

    def adjust_zoom(self, view):
        if view == "xy":

            self.zoom_level_xy = self.xy_zoom_slider.value() / 100.0
        elif view == "xz":
            self.zoom_level_xz = self.xz_zoom_slider.value() / 100.0
        elif view == "yz":
            self.zoom_level_yz = self.zy_zoom_slider.value() / 100.0

    def update_slice(self, view):

        slider = self.get_slider(view)
        xy_idx = self.xy_slider.value()
        xz_idx = self.xz_slider.value()
        zy_idx = self.zy_slider.value()

        self.update_xy_image(xy_idx)
        self.update_xz_image(xz_idx)
        self.update_zy_image(zy_idx)
        # Increment the slider value by 1
        current_value = slider.value()
        if current_value < slider.maximum() - 1:
            slider.setValue(current_value + 1)  # Increment slider by 1
        else:
            slider.setValue(0)  # Reset to the first slice when reaching the last

    def get_slider(self, view):
        """
        Return the correct slider for the given view.
        """
        if view == "xy":
            return self.xy_slider
        elif view == "xz":
            return self.xz_slider
        else:
            return self.zy_slider

    def initUI(self):
        # Labels for displaying images with fixed size
        self.xy_label = QLabel(self)
        self.xy_label.setAlignment(Qt.AlignCenter)
        self.xy_label.setMinimumSize(200, 200)
        self.xy_label.setMaximumHeight(300)
        self.xy_label.setStyleSheet("border: 1px solid rgb(9, 132, 227);; background-color: black;")
        # Set minimum size for XY view
        self.xy_label.mousePressEvent = self.mouse_press_event_xy
        self.xy_label.setMouseTracking(True)
        self.xy_label.mouseMoveEvent = self.mouse_move_event_xy

        self.xz_label = QLabel(self)
        self.xz_label.setAlignment(Qt.AlignCenter)
        self.xz_label.setMinimumSize(200, 200)  # Set minimum size for XZ view
        self.xz_label.setMaximumHeight(300)
        self.xz_label.setStyleSheet("border: 1px solid rgb(9, 132, 227);; background-color: black;")
        self.xz_label.mousePressEvent = self.mouse_press_event_xz
        self.xz_label.setMouseTracking(True)
        self.xz_label.mouseMoveEvent = self.mouse_move_event_xz

        self.zy_label = QLabel(self)
        self.zy_label.setAlignment(Qt.AlignCenter)
        self.zy_label.setMinimumSize(200, 200)  # Set minimum size for YZ view
        self.zy_label.setMaximumHeight(300)
        self.zy_label.setStyleSheet("border: 1px solid rgb(9, 132, 227);; background-color: black;")
        self.zy_label.mousePressEvent = self.mouse_press_event_zy
        self.zy_label.setMouseTracking(True)
        self.zy_label.mouseMoveEvent = self.mouse_move_event_zy

        # Sliders for navigating slices
        self.xy_slider = QSlider(Qt.Horizontal, self)
        self.xz_slider = QSlider(Qt.Horizontal, self)
        self.zy_slider = QSlider(Qt.Horizontal, self)

        # Brightness, Contrast, and Zoom sliders
        self.xy_brightness_slider = QSlider(Qt.Horizontal, self)
        self.xy_brightness_slider.setRange(10, 300)
        self.xy_brightness_slider.valueChanged.connect(lambda value, v="xy": self.update_brightness(value, v))
        # self.xy_brightness_slider["xy"] = self.xy_brightness_slider
        self.xz_brightness_slider = QSlider(Qt.Horizontal, self)

        self.xz_brightness_slider.setRange(10, 300)
        self.xz_brightness_slider.valueChanged.connect(lambda value, v="xz": self.update_brightness(value, v))
        self.zy_brightness_slider = QSlider(Qt.Horizontal, self)
        self.zy_brightness_slider.setRange(10, 300)
        self.zy_brightness_slider.valueChanged.connect(lambda value, v="zy": self.update_brightness(value, v))

        self.xy_contrast_slider = QSlider(Qt.Horizontal, self)
        self.xy_contrast_slider.setRange(0, 100)  # Scale from 1x to 3x
        self.xy_contrast_slider.valueChanged.connect(lambda value, v="xy": self.update_contrast(value, v))

        self.xz_contrast_slider = QSlider(Qt.Horizontal, self)
        self.xz_contrast_slider.setRange(0, 100)  # Scale from 1x to 3x
        self.xz_contrast_slider.valueChanged.connect(lambda value, v="xz": self.update_contrast(value, v))

        self.zy_contrast_slider = QSlider(Qt.Horizontal, self)
        self.zy_contrast_slider.setRange(0, 100)  # Scale from 1x to 3x
        self.zy_contrast_slider.valueChanged.connect(lambda value, v="zy": self.update_contrast(value, v))

        self.xy_zoom_slider = QSlider(Qt.Horizontal, self)
        self.xz_zoom_slider = QSlider(Qt.Horizontal, self)
        self.zy_zoom_slider = QSlider(Qt.Horizontal, self)

        # Create main layout
        main_layout = QHBoxLayout()

        # Create control area (group box) on the left
        control_area = QGroupBox("Control Area")
        control_area.setFixedWidth(200)  # Set fixed width for control area
        control_layout = QVBoxLayout()

        # Control sliders for XY view
        control_layout.addWidget(QLabel("Axial (XY) View"))
        control_layout.addWidget(QLabel("Brightness"))
        self.xy_brightness_slider.setMinimumSize(150, 30)  # Set minimum size for sliders
        control_layout.addWidget(self.xy_brightness_slider)
        control_layout.addWidget(QLabel("Contrast"))
        control_layout.addWidget(self.xy_contrast_slider)
        control_layout.addWidget(QLabel("Zoom"))
        control_layout.addWidget(self.xy_zoom_slider)
        self.xy_zoom_slider.setRange(40, 200)  # 10% to 200%
        self.xy_zoom_slider.setValue(40)  # Start at 100%
        self.xy_zoom_slider.valueChanged.connect(lambda: self.adjust_zoom(("xy")))
        control_layout.addWidget(QLabel("Slice"))
        control_layout.addWidget(self.xy_slider)

        self.xy_play_button = QPushButton()
        self.xy_play_button.setIcon(self.create_play_icon())  # Use custom green play icon
        self.xy_play_button.setIconSize(QSize(30, 20))  # Set icon size
        self.xy_play_button.setFixedHeight(25)
        self.xy_play_button.clicked.connect(lambda: self.toggle_play(("xy")))
        control_layout.addWidget(self.xy_play_button)
        # Add separator line
        line_xy = QFrame()
        line_xy.setFrameShape(QFrame.HLine)  # Horizontal line
        line_xy.setFrameShadow(QFrame.Sunken)
        line_xy.setStyleSheet(" background-color: rgb(9, 132, 227);")  # Set line color and background
        control_layout.addWidget(line_xy)

        # Control sliders for XZ view
        control_layout.addWidget(QLabel("Coronal (XZ) View"))
        control_layout.addWidget(QLabel("Brightness"))
        control_layout.addWidget(self.xz_brightness_slider)
        control_layout.addWidget(QLabel("Contrast"))
        control_layout.addWidget(self.xz_contrast_slider)
        control_layout.addWidget(QLabel("Zoom"))
        control_layout.addWidget(self.xz_zoom_slider)
        self.xz_zoom_slider.setRange(40, 200)  # 10% to 200%
        self.xz_zoom_slider.setValue(40)  # Start at 100%
        self.xz_zoom_slider.valueChanged.connect(lambda: self.adjust_zoom(("xz")))
        control_layout.addWidget(QLabel("Slice"))
        control_layout.addWidget(self.xz_slider)
        self.xz_play_button = QPushButton()
        self.xz_play_button.setIcon(self.create_play_icon())  # Use custom green play icon
        self.xz_play_button.setIconSize(QSize(30, 20))  # Set icon size
        self.xz_play_button.setFixedHeight(25)
        self.xz_play_button.clicked.connect(lambda: self.toggle_play(("xz")))  # Connect button to toggle function
        control_layout.addWidget(self.xz_play_button)
        # Add separator line
        line_xz = QFrame()
        line_xz.setFrameShape(QFrame.HLine)  # Horizontal line
        line_xz.setFrameShadow(QFrame.Sunken)
        line_xz.setStyleSheet("background-color: rgb(9, 132, 227);")  # Set line color and background
        control_layout.addWidget(line_xz)

        # Control sliders for YZ view
        control_layout.addWidget(QLabel("Sagittal (YZ) View"))
        control_layout.addWidget(QLabel("Brightness"))
        control_layout.addWidget(self.zy_brightness_slider)
        control_layout.addWidget(QLabel("Contrast"))
        control_layout.addWidget(self.zy_contrast_slider)
        control_layout.addWidget(QLabel("Zoom"))
        control_layout.addWidget(self.zy_zoom_slider)
        control_layout.addWidget(QLabel("Slice"))
        control_layout.addWidget(self.zy_slider)
        self.zy_zoom_slider.setRange(40, 200)  # 10% to 200%
        self.zy_zoom_slider.setValue(40)  # Start at 100%
        self.zy_zoom_slider.valueChanged.connect(lambda: self.adjust_zoom(("yz")))
        self.yz_play_button = QPushButton()
        self.yz_play_button.setIcon(self.create_play_icon())  # Use custom green play icon
        self.yz_play_button.setIconSize(QSize(30, 20))  # Set icon size
        self.yz_play_button.setFixedHeight(25)
        self.yz_play_button.clicked.connect(lambda: self.toggle_play(("yz")))
        control_layout.addWidget(self.yz_play_button)
        # Set the layout for the control area
        control_area.setLayout(control_layout)
        # Add the control area and the views to the main layout
        main_layout.addWidget(control_area)
        # Create a layout for the views
        views_layout = QVBoxLayout()
        views_layout.addWidget(QLabel("Axial (XY) View"))
        views_layout.addWidget(self.xy_label)
        views_layout.addWidget(QLabel("Coronal (XZ) View"))
        views_layout.addWidget(self.xz_label)
        views_layout.addWidget(QLabel("Sagittal (YZ) View"))
        views_layout.addWidget(self.zy_label)

        # Add the views layout to the main layout
        main_layout.addLayout(views_layout)

        # Set the main layout as the window's layout
        self.setLayout(main_layout)

    def update_brightness(self, value, view):
        self.contrast[view] = value / 100.0  # Scale to usable range
        self.update_all_images()  # Refresh images

    def update_contrast(self, value, view):
        self.brightness[view] = value
        self.update_all_images()  # Refresh images

    def toggle_play(self, view):
        # Toggle play/pause for the specified view
        if view == "xy":
            self.playing_state["xy"] = not self.playing_state["xy"]

            if self.playing_state["xy"]:
                self.timers["xy"].start(2)
                self.playing_state["xy"] = True
                self.xy_play_button.setIcon(self.create_pause_icon())
            else:
                self.timers["xy"].stop()
                self.playing_state["xy"] = False
                self.xy_play_button.setIcon(self.create_play_icon())

            # Logic to start/stop playing XY view slices goes here
        elif view == "xz":
            self.playing_state["xz"] = not self.playing_state["xz"]

            if self.playing_state["xz"]:
                self.timers["xz"].start(2)
                self.playing_state["xz"] = True
                self.xz_play_button.setIcon(self.create_pause_icon())

            else:
                self.timers["xz"].stop()
                self.playing_state["xz"] = False
                self.xz_play_button.setIcon(self.create_play_icon())

        elif view == "yz":
            self.playing_state["yz"] = not self.playing_state["yz"]
            if self.playing_state["yz"]:
                self.timers["yz"].start(2)
                self.playing_state["yz"] = True
                self.yz_play_button.setIcon(self.create_pause_icon())

            else:
                self.timers["yz"].stop()
                self.playing_state["yz"] = False
                self.yz_play_button.setIcon(self.create_play_icon())

    def load_image(self, image_array):
        """
        Load the image array and initialize sliders.
        """
        self.image_array = image_array
        max_axial = self.image_array.shape[0] - 1
        max_sagittal = self.image_array.shape[1] - 1
        max_coronal = self.image_array.shape[2] - 1

        self.xy_slider.setMaximum(max_axial)
        self.xz_slider.setMaximum(max_sagittal)
        self.zy_slider.setMaximum(max_coronal)

        # Set initial slider positions to the middle slices
        self.xy_slider.setValue(max_axial // 2)
        self.xz_slider.setValue(max_sagittal // 2)
        self.zy_slider.setValue(max_coronal // 2)
        self.current_crosshair = None
        self.update_all_images()

    def update_all_images(self):
        """
        Update all image views based on current slider values.
        """
        xy_idx = self.xy_slider.value()
        xz_idx = self.xz_slider.value()
        zy_idx = self.zy_slider.value()

        self.update_xy_image(xy_idx)
        self.update_xz_image(xz_idx)
        self.update_zy_image(zy_idx)

    def update_xy_image(self, slice_idx):
        if self.image_array is not None:
            slice_image = self.image_array[slice_idx, :, :]
            # Apply 180-degree rotation (flipping vertically and horizontally)
            slice_image = np.flipud(np.fliplr(slice_image))
            # Display the adjusted image
            self.display_image(slice_image, self.xy_label, plane='xy', zoom_factor=self.zoom_level_xy)

    def update_xz_image(self, slice_idx):
        if self.image_array is not None:
            slice_image = self.image_array[:, slice_idx, :]

            # Apply 180-degree rotation (flipping vertically and horizontally)
            slice_image = np.flipud(np.fliplr(slice_image))

            # Display the adjusted image
            self.display_image(slice_image, self.xz_label, plane='xz', zoom_factor=self.zoom_level_xz)

    def update_zy_image(self, slice_idx):
        if self.image_array is not None:
            slice_image = self.image_array[:, :, slice_idx]
            # Apply 180-degree rotation (flipping vertically and horizontally)
            slice_image = np.flipud(np.fliplr(slice_image))

            # Display the adjusted image
            self.display_image(slice_image, self.zy_label, plane='zy', zoom_factor=self.zoom_level_yz)

    def display_image(self, slice_image, label, plane, zoom_factor=0.4):
        height, width = slice_image.shape
        min_val = np.min(slice_image)
        max_val = np.max(slice_image)

        # Normalize the image
        if max_val != min_val:
            slice_image = (slice_image - min_val) / (max_val - min_val) * 255
        else:
            slice_image = np.zeros_like(slice_image)

        # Apply brightness and contrast adjustments
        brightness = self.brightness[plane]
        contrast = self.contrast[plane]
        slice_image = np.clip(slice_image * contrast + brightness, 0, 255).astype(np.uint8)
        q_image = QImage(slice_image.data, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        # Draw crosshair if the current position is set
        if self.current_crosshair is not None:
            painter = QPainter(pixmap)
            painter.setPen(QColor(255, 0, 0))  # Red color for crosshair
            crosshair_x, crosshair_y = self.current_crosshair

            # Scale the pixmap to match the label size
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_width = scaled_pixmap.width()
            scaled_height = scaled_pixmap.height()

            # Calculate offset for centered image in label
            offset_x = (label.width() - scaled_width) // 2
            offset_y = (label.height() - scaled_height) // 2

            # Map cursor crosshair coordinates to scaled image dimensions
            scale_x = width / scaled_width if scaled_width != 0 else 1
            scale_y = height / scaled_height if scaled_height != 0 else 1

            # Adjust crosshair coordinates
            crosshair_x = int((crosshair_x - offset_x) * scale_x)
            crosshair_y = int((crosshair_y - offset_y) * scale_y)

            # Ensure crosshair is drawn within bounds
            if 0 <= crosshair_x < pixmap.width() and 0 <= crosshair_y < pixmap.height():
                # Draw crosshair lines
                painter.drawLine(crosshair_x, 0, crosshair_x, pixmap.height())  # Vertical line
                painter.drawLine(0, crosshair_y, pixmap.width(), crosshair_y)  # Horizontal line

                # Draw small circle at the crosshair position
                painter.setPen(QColor(0, 255, 0))  # Green color for
                painter.drawEllipse(crosshair_x - 5, crosshair_y - 5, 10, 10)  # Small circle displaying on crosshair
                # painter.setBrush(QColor(0, 255, 0))  # Set brush color for filling dots

        # Scale the pixmap again for displaying in the label
        scaled_pixmap = pixmap.scaled(pixmap.size() * zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Set the scaled pixmap in the label
        label.setPixmap(scaled_pixmap)

        return pixmap

    def update_crosshair(self):
        """
        Update crosshair positions across all views.
        """
        if self.current_crosshair is None:
            return

        self.update_all_images()

    def mouse_move_event_xy(self, event):
        """
        Handle mouse move in XY view.
        """
        if self.xy_label.rect().contains(event.pos()):
            self.current_crosshair = (event.pos().x(), event.pos().y())
            self.update_all_images()

    def mouse_move_event_xz(self, event):
        """
        Handle mouse move in XZ view.
        """
        if self.xz_label.rect().contains(event.pos()):
            self.current_crosshair = (event.pos().x(), event.pos().y())
            self.update_all_images()

    def mouse_move_event_zy(self, event):
        """
        Handle mouse move in ZY view.
        """
        if self.zy_label.rect().contains(event.pos()):
            self.current_crosshair = (event.pos().x(), event.pos().y())
            self.update_all_images()

    def mouse_press_event_xy(self, event):
        """
        Handle mouse click in XY view.
        """
        max_coronal = self.image_array.shape[2] - 1
        max_sagittal = self.image_array.shape[1] - 1

        if self.xy_label.rect().contains(event.pos()) and self.image_array is not None:
            # Get the size of the pixmap to determine scaling
            current_pixmap = self.xy_label.pixmap()
            if current_pixmap is None:
                return  # Exit if there's no pixmap available

            scaled_pixmap = current_pixmap.scaled(self.xy_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Get the size of the scaled pixmap
            scaled_width = scaled_pixmap.width()
            scaled_height = scaled_pixmap.height()

            # Calculate offset for centering the image
            offset_x = (self.xy_label.width() - scaled_width) // 2
            offset_y = (self.xy_label.height() - scaled_height) // 2

            # Get mouse coordinates
            x = event.pos().x() - offset_x
            y = event.pos().y() - offset_y

            # Ensure that x and y are within bounds of the scaled image
            if 0 <= x < scaled_width and 0 <= y < scaled_height:
                # Map clicked point to image indices
                image_x = int((x / scaled_width) * self.image_array.shape[2])
                image_y = int((y / scaled_height) * self.image_array.shape[1])

                # Update other sliders based on clicked position
                self.xz_slider.setValue(max_sagittal - image_y)
                self.zy_slider.setValue(max_coronal - image_x)

                # Update crosshair
                self.current_crosshair = (x + offset_x, y + offset_y)  # Store original position for crosshair
                self.update_all_images()

    def mouse_press_event_zy(self, event):
        """
        Handle mouse click in ZY view.
        """
        max_axial = self.image_array.shape[0] - 1
        max_sagittal = self.image_array.shape[1] - 1

        if self.zy_label.rect().contains(event.pos()) and self.image_array is not None:
            # Get the size of the pixmap to determine scaling
            current_pixmap = self.zy_label.pixmap()
            if current_pixmap is None:
                return  # Exit if there's no pixmap available

            scaled_pixmap = current_pixmap.scaled(self.zy_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Get the size of the scaled pixmap
            scaled_width = scaled_pixmap.width()
            scaled_height = scaled_pixmap.height()

            # Calculate offset for centering the image
            offset_x = (self.zy_label.width() - scaled_width) // 2
            offset_y = (self.zy_label.height() - scaled_height) // 2

            # Get mouse coordinates
            x = event.pos().x() - offset_x
            y = event.pos().y() - offset_y

            # Ensure that x and y are within bounds of the scaled image
            if 0 <= x < scaled_width and 0 <= y < scaled_height:
                # Map clicked point to image indices
                image_x = int((x / scaled_width) * self.image_array.shape[1])
                image_y = int((y / scaled_height) * self.image_array.shape[0])

                # Update other sliders based on clicked position
                self.xy_slider.setValue(max_axial - image_y)
                self.xz_slider.setValue(max_sagittal - image_x)

                # Update crosshair
                self.current_crosshair = (x + offset_x, y + offset_y)  # Store original position for crosshair

                self.update_all_images()
        else:
            # Handle left-click or other behavior
            pass

    def mouse_press_event_xz(self, event):
        """
        Handle mouse click in XZ view.
        """
        max_axial = self.image_array.shape[0] - 1
        max_coronal = self.image_array.shape[2] - 1

        if self.xz_label.rect().contains(event.pos()) and self.image_array is not None:
            # Get the size of the pixmap to determine scaling
            current_pixmap = self.xz_label.pixmap()
            if current_pixmap is None:
                return  # Exit if there's no pixmap available

            scaled_pixmap = current_pixmap.scaled(self.xz_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Get the size of the scaled pixmap
            scaled_width = scaled_pixmap.width()
            scaled_height = scaled_pixmap.height()

            # Calculate offset for centering the image
            offset_x = (self.xz_label.width() - scaled_width) // 2
            offset_y = (self.xz_label.height() - scaled_height) // 2

            # Get mouse coordinates
            x = event.pos().x() - offset_x
            y = event.pos().y() - offset_y

            # Ensure that x and y are within bounds of the scaled image
            if 0 <= x < scaled_width and 0 <= y < scaled_height:
                # Map clicked point to image indices
                image_x = int((x / scaled_width) * self.image_array.shape[2])
                image_y = int((y / scaled_height) * self.image_array.shape[0])

                # Update other sliders based on clicked position
                self.xy_slider.setValue(max_axial - image_y)
                self.zy_slider.setValue(max_coronal - image_x)

                # Update crosshair
                self.current_crosshair = (x + offset_x, y + offset_y)  # Store original position for crosshair
                self.update_all_images()


class VolumeRenderer(QWidget):
    """
    3D Volume Rendering using VTK.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize renderer and related attributes before calling initUI
        self.renderer = vtk.vtkRenderer()
        self.volume = None
        self.volume_mapper = None
        self.volume_property = vtk.vtkVolumeProperty()
        self.color_transfer_function = vtk.vtkColorTransferFunction()
        self.opacity_transfer_function = vtk.vtkPiecewiseFunction()
        self.image_importer = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        # Add renderer to the render window
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Setup interactor style (optional)
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(interactor_style)

        self.setLayout(layout)

    def load_image(self, image_array):
        """
        Load the image array and setup VTK volume rendering.
        """
        if image_array is None:
            QMessageBox.warning(self, "No Image Data", "No image data to render.")
            return

        # Clear previous renderer
        self.renderer.RemoveAllViewProps()

        # Determine the data type
        if image_array.dtype == np.uint8:
            vtk_data_type = vtk.VTK_UNSIGNED_CHAR
        elif image_array.dtype == np.uint16:
            vtk_data_type = vtk.VTK_UNSIGNED_SHORT
        elif image_array.dtype == np.float32:
            vtk_data_type = vtk.VTK_FLOAT
        elif image_array.dtype == np.float64:
            vtk_data_type = vtk.VTK_DOUBLE
        else:
            QMessageBox.critical(self, "Unsupported Data Type",
                                 f"Data type {image_array.dtype} is not supported.")
            return

        # Convert NumPy array to VTK image data using vtkImageImport
        self.image_importer = vtk.vtkImageImport()
        data_string = image_array.tobytes()
        self.image_importer.CopyImportVoidPointer(data_string, len(data_string))
        self.image_importer.SetDataScalarType(vtk_data_type)
        self.image_importer.SetNumberOfScalarComponents(1)  # Grayscale
        self.image_importer.SetDataExtent(0, image_array.shape[2] - 1,
                                          0, image_array.shape[1] - 1,
                                          0, image_array.shape[0] - 1)
        self.image_importer.SetWholeExtent(0, image_array.shape[2] - 1,
                                           0, image_array.shape[1] - 1,
                                           0, image_array.shape[0] - 1)
        self.image_importer.SetDataSpacing(1, 1, 1)  # Adjust if spacing is known
        self.image_importer.Update()

        # Setup Volume Mapper
        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        self.volume_mapper.SetInputConnection(self.image_importer.GetOutputPort())

        # Setup Transfer Functions
        self.initialize_transfer_functions(image_array)

        # Setup Volume Property
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.color_transfer_function)
        self.volume_property.SetScalarOpacity(self.opacity_transfer_function)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()

        # Setup Volume
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        self.renderer.AddVolume(self.volume)

        # Setup Renderer
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Azimuth(30)
        self.renderer.GetActiveCamera().Elevation(30)
        self.renderer.ResetCameraClippingRange()

        # Render
        self.vtk_widget.GetRenderWindow().Render()

    def initialize_transfer_functions(self, image_array):
        """
        Initialize default color and opacity transfer functions.
        """
        self.color_transfer_function.RemoveAllPoints()
        self.opacity_transfer_function.RemoveAllPoints()

        data_range = image_array.min(), image_array.max()
        min_val, max_val = data_range

        # Color Transfer Function
        self.color_transfer_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
        self.color_transfer_function.AddRGBPoint(max_val * 0.25, 0.85, 0.55, 0.3)
        self.color_transfer_function.AddRGBPoint(max_val * 0.5, 0.95, 0.85, 0.7)
        self.color_transfer_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0)

        # Opacity Transfer Function
        self.opacity_transfer_function.AddPoint(min_val, 0.0)
        self.opacity_transfer_function.AddPoint(max_val * 0.25, 0.1)
        self.opacity_transfer_function.AddPoint(max_val * 0.5, 0.4)
        self.opacity_transfer_function.AddPoint(max_val, 1.0)


class MainWindow(QMainWindow):
    """
    Main Application Window Combining 2D Viewer and 3D Renderer.
    """

    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_loader_thread = None

    def initUI(self):
        self.setWindowTitle("Team ##9 advanced-3d-image-basing")
        self.setWindowIcon(QIcon('icon.ico'))
        self.setGeometry(0, 0, 800, 600)

        # Initialize Toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        # Load Image Button
        self.load_button = QPushButton("Load Image")
        self.load_button.setToolTip("Load a NIfTI image")
        self.load_button.clicked.connect(self.load_image)
        self.toolbar.addWidget(self.load_button)

        # Initialize Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        status_separator = QFrame()
        status_separator.setFrameShape(QFrame.HLine)  # Vertical line
        status_separator.setFrameShadow(QFrame.Sunken)
        status_separator.setStyleSheet("color: rgb(9, 132, 227); background-color: rgb(9, 132, 227);")

        self.status_bar.addPermanentWidget(status_separator)
        self.status_bar.setStyleSheet("background-color: rgb(9, 132, 227);")
        # Initialize Tabs
        self.tabs = QTabWidget()
        self.image_viewer = ImageViewer()
        self.volume_renderer = VolumeRenderer()
        self.tabs.addTab(self.image_viewer, "2D Viewer")
        self.tabs.currentChanged.connect(self.change_tab_color)
        self.tabs.addTab(self.volume_renderer, "3D Renderer")
        self.setCentralWidget(self.tabs)

    def load_image(self):
        """
        Open file dialog to select and load NIfTI image.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image File", "", "NIFTI Files (*.nii *.nii.gz)"
        )
        if file_path:
            self.status_bar.showMessage("Loading image...")
            self.load_button.setEnabled(False)

            # Initialize and start image loader thread
            self.image_loader_thread = ImageLoaderThread(file_path)
            self.image_loader_thread.image_loaded.connect(self.on_image_loaded)
            self.image_loader_thread.error_occurred.connect(self.on_load_error)
            self.image_loader_thread.start()

    def on_image_loaded(self, image_array, itk_image):
        """
        Handle the loaded image data.
        """
        self.status_bar.showMessage("Image loaded successfully", 2000)
        self.load_button.setEnabled(True)

        # Load image into 2D Viewer
        self.image_viewer.load_image(image_array)

        # Load image into 3D Renderer
        self.volume_renderer.load_image(image_array)

    def on_load_error(self, error_message):
        """
        Handle errors during image loading.
        """
        self.status_bar.showMessage("Error loading image", 2000)
        self.load_button.setEnabled(True)
        QMessageBox.critical(self, "Load Error", f"An error occurred while loading the image:\n{error_message}")

    def change_tab_color(self, index):
        # Change the color of the active tab based on the current index
        for i in range(self.tabs.count()):
            if i == index:
                self.tabs.tabBar().setTabTextColor(i, Qt.cyan)  # Change text color of the active tab

            else:
                self.tabs.tabBar().setTabTextColor(i, Qt.darkGray)  # Reset text color of inactive tabs


stylesheet = """ 
QWidget{ background-color: rgb(30,30,30);color: White;}
QLabel{ color: White;}
QPushButton {color: White; }
QTabWidget  {color: White; }
"""


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(stylesheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
