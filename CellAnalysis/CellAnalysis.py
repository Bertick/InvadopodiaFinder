import sys
import os.path

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from skimage.segmentation import watershed
from skimage.draw import line
import scipy.ndimage as ndi

from PyQt5.QtCore import QSettings, pyqtSlot, Qt
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from pyqtgraph import BusyCursor, ROI

from .Gui.mainwindow import Ui_MainWindow

import logging

VIEW_OPTIONS = {
    'Segmented image': 'image_overlay',
    'Input image': 'image',
}
# note: I'm pretty sure that if BACKGROUND_LABEL is different from 0, stuff will break
BACKGROUND_LABEL = 0
THRESHOLD_COLOR = (0.0, 0.0, 1.0)
# marker colors
mACCEPT_COLORS = [
    (0.75, 0.75, 0),
    (0, 0.75, 0),
    (0, 0.75, 0.75),
]
mIGNORE_COLORS = [
    (0.25, 0, 0)
]
mCROSS_COLORS = [
    (0.0, 1.0, 0.0),  # accept
    (1.0, 0.0, 0.0),  # ignore
]
# marker actions
mACTION_ADD = 'add'
mACTION_REMOVE = 'remove'
mACTION_IGNORE = 'ignore'


class InvadopodiaGui(QMainWindow, Ui_MainWindow):
    """
    Class defines the Gui and program logic, following the 'smart GUI' approach
    """
    def __init__(self, parent=None):
        super(InvadopodiaGui, self).__init__(parent=parent)

        # setup the UI code
        self.setupUi(self)
        self.read_settings()

        # data container
        self.data = {}

        # register options in comboBox
        for key in VIEW_OPTIONS.keys():
            self.combobox_viewoptions.addItem(key)

        # remove ROI and norm buttons from ImageView
        self.imageView.ui.roiBtn.hide()
        self.imageView.ui.menuBtn.hide()

        # slots
        # load image
        self.action_load_image.triggered.connect(self._on_load_image)
        # view change
        self.combobox_viewoptions.currentTextChanged.connect(self._update_imageview)
        # thresholds
        self.bttn_compute_otsu.pressed.connect(self._compute_threshold)
        self.box_threshold.valueChanged.connect(self._update_images)
        # mouseClicked event from ImageView scene
        self.imageView.scene.sigMouseClicked.connect(self._on_mouse_click)
        # Marker buttons toggle
        self.bttn_addmarker.toggled.connect(self._on_markerbuttons_click)
        self.bttn_removemarker.toggled.connect(self._on_markerbuttons_click)
        self.bttn_ignoremarker.toggled.connect(self._on_markerbuttons_click)
        # write report
        self.bttn_dosave.released.connect(self._write_report)
        # ROI button toggle
        self.bttn_toggle_roi.toggled.connect(self._toggle_roi)
        # logging
        logging.debug(f'Created InvadopodiaGui - {id(self)}')

    #############################
    # CALLBACKS FOR USER ACTION #
    #############################
    @pyqtSlot(bool)
    def _on_load_image(self, checked):
        fname, _ = QFileDialog.getOpenFileName(
            None,
            'Open file',
            os.path.expanduser('~'),
            'Images (*.tif *.tiff *.jpg *.jpeg *.png);; All files (*.*)'
        )
        if not len(fname):
            return
        #
        try:
            image = imread(fname, as_gray=True)
            # image = rgb2gray(image)
            image = np.uint8(image * 255)
        except (FileNotFoundError, FileExistsError) as e:
            QMessageBox.warning(self, 'Whoops!', 'Error: File not found', QMessageBox.Ok)
            logging.warning(e)
            return
        except PermissionError as e:
            QMessageBox.warning(self, 'Whoops!', "Error: Permission denied", QMessageBox.Ok)
            logging.warning(e)
            return
        except Exception as e:
            QMessageBox.warning(self, 'Whoops!', "Unknown error occurred: program will close", QMessageBox.Ok)
            logging.error(e)
            sys.exit(1)
        else:
            self.data['image'] = image
            self.data['roi'] = None
            self.data['image_overlay'] = np.zeros_like(image)
            self.data['image_markers'] = np.zeros_like(image)
            self.data['image_watershed'] = np.zeros_like(image)
            self.data['new_marker'] = ()
            self.data['lb_act_map'] = {}
            self._compute_threshold()
            self._update_imageview(autoLevels=True, autoRange=True)

    @pyqtSlot()
    def _compute_threshold(self):
        try:
            image = self.data['image']
        except KeyError:
            return
        threshold = threshold_otsu(image)
        logging.info(f'Threshold computed: {threshold:.3f}')
        self.box_threshold.blockSignals(True)
        self.box_threshold.setValue(threshold)
        self.box_threshold.blockSignals(False)
        self._update_images()

    def _on_mouse_click(self, event):
        # if event is already accepted (when? why?) we ignore it
        if event.isAccepted():
            return
        # check that it's the left mouse button
        if event.buttons() != Qt.LeftButton:
            return
        # get event location in image coordinates
        pos = self.imageView.view.mapSceneToView(event.scenePos())
        pos = (int(pos.x()), int(pos.y()))
        logging.debug(f'Mouse click at {pos}')

        if self.bttn_addmarker.isChecked():
            action = mACTION_ADD
        elif self.bttn_removemarker.isChecked():
            action = mACTION_REMOVE
        elif self.bttn_ignoremarker.isChecked():
            action = mACTION_IGNORE
        else:
            return
        self.data['new_marker'] = (pos, action)
        event.accept()
        self._process_new_marker()

    @pyqtSlot(bool)
    def _on_markerbuttons_click(self, checked):
        if not checked:
            return
        sender = self.sender()
        if sender is self.bttn_addmarker:
            self.bttn_removemarker.setChecked(False)
            self.bttn_ignoremarker.setChecked(False)
        elif sender is self.bttn_removemarker:
            self.bttn_addmarker.setChecked(False)
            self.bttn_ignoremarker.setChecked(False)
        else:
            self.bttn_addmarker.setChecked(False)
            self.bttn_removemarker.setChecked(False)

    def _write_report(self):
        try:
            image = self.data['image']
            image_watershed = self.data['image_watershed']
            label_to_action = self.data['lb_act_map']
        except KeyError:
            return
        scale = float(self.doublespinbox_pixeltoum.value())
        # warn the user if scale is zero
        if not scale:
            ans = QMessageBox.question(
                self,
                'Think carefully',
                "'Size' parameter is zero or None. Are you sure you want to continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            if ans == QMessageBox.Cancel:
                return

        # get where user wants to save files
        fname, _ = QFileDialog.getSaveFileName(
            self,
            'Save Report',
            os.path.expanduser('~'),
        )
        if not fname:
            return
        # remove file extension (if present)
        fname, _ = os.path.splitext(fname)
        # remove all ignored labels
        for i, action in label_to_action.items():
            if action == mACTION_IGNORE:
                image_watershed[image_watershed == i] = BACKGROUND_LABEL

        # compute region properties
        props = regionprops_table(
            image_watershed,
            intensity_image=image,
            # todo: move these to an option menu
            properties=('label', 'area', 'centroid', 'eccentricity', 'mean_intensity')
        )

        df = pd.DataFrame(props)
        df['area'] *= scale**2

        df_summary = df.describe()
        # save to excel files
        df.to_excel(fname + '.xlsx')
        df_summary.to_excel(fname + '_summary.xlsx')
        logging.info(f'Report written to {fname}')

    #############
    # INTERNALS #
    #############
    def _process_new_marker(self):
        try:
            image_th = self.data['image_th']
            image_markers = self.data['image_markers']
            image_watershed = self.data['image_watershed']
        except KeyError:
            return
        # get index <-> action map
        try:
            label_to_action = self.data['lb_act_map']
        except KeyError:
            label_to_action = {}
            self.data['lb_act_map'] = label_to_action
        # check if there is a new marker from user action
        try:
            pos, action = self.data['new_marker']
            self.data['new_marker'] = None
        except (KeyError, TypeError):
            return
        # ignore markers that fall on the threshold area
        if image_th[pos] == BACKGROUND_LABEL:
            return

        logging.debug(f'Processing marker at {pos}, with action {action}')
        # operate based on action
        if action == mACTION_ADD or action == mACTION_IGNORE:
            label = image_markers.max() + 1
            image_markers[pos] = label
            label_to_action[label] = action
        elif action == mACTION_REMOVE:
            label = image_watershed[pos]
            if label != BACKGROUND_LABEL:
                # removes any marker present in the clicked region
                image_markers[image_watershed == label] = BACKGROUND_LABEL

            label_to_action.pop(label, None)
        else:
            logging.error(f'Marker at {pos} has unknown action {action}. Raising RuntimeError')
            raise RuntimeError(f'Unknown marker action {action}')
        # update the images
        self._update_images()

    def _update_images(self):
        with BusyCursor():
            try:
                image = self.data['image']
                image_markers = self.data['image_markers']
            except KeyError:
                # some data (like image) is not defined/loaded
                return

            try:
                roi, _ = self.data['roi'].getArraySlice(image, self.imageView.imageItem)
            except AttributeError:
                # self.data['roi'] is None
                roi = None

            # first check if some markers are outside the ROI position and remove them
            if roi:
                mask = np.ones_like(image_markers)
                mask[roi] = 0

                for cc in zip(*image_markers.nonzero()):
                    label = image_markers[cc]
                    if mask[cc]:
                        # marker is outside ROI
                        image_markers[cc] = 0
                        self.data['lb_act_map'].pop(label, None)

                # reduce image size to ROI
                image = image[roi]
                image_markers = image_markers[roi]

            th = float(self.box_threshold.value())
            # compute threshold overlay
            image_th = _compute_threshold_image(image, th)

            if image_markers.any():
                image_watershed = _compute_watershed(image_th, image_markers)
            else:
                # no markers are present
                image_watershed = np.zeros_like(image)

            # compute overlay
            # todo: soft-code alpha
            colors = []
            try:
                label_to_action = self.data['lb_act_map']
                for i, action in label_to_action.items():
                    if action == mACTION_ADD:
                        color = mACCEPT_COLORS[i % len(mACCEPT_COLORS)]
                    else:
                        color = mIGNORE_COLORS[i % len(mIGNORE_COLORS)]
                    colors.append(color)
            except KeyError:
                pass
            # add the color for threshold areas
            colors.append(THRESHOLD_COLOR)
            # to color the threshold(ed) area (i.e. the background) a different color, we assign a dummy label (-1)
            tmp = np.zeros_like(image)
            tmp[image_th == BACKGROUND_LABEL] = -1

            # todo: soft-code alpha
            image_overlay = label2rgb(
                tmp + image_watershed,
                image,
                alpha=0.4,
                bg_label=BACKGROUND_LABEL,
                bg_color=None,
                colors=colors
            )

            # draw markers
            for cc in zip(*image_markers.nonzero()):
                label = image_markers[cc]
                action = label_to_action[label]
                color = mCROSS_COLORS[0] if action == mACTION_ADD else mCROSS_COLORS[1]
                # todo: soft-code size
                _draw_cross(image_overlay, cc, 20, color)

            if roi:
                # restore images to the proper size
                tmp = np.zeros(shape=self.imageView.image.shape[0:2], dtype=image_watershed.dtype)
                tmp[roi] = image_watershed
                image_watershed = tmp

                tmp = np.stack([self.data['image'] / 255 for _ in range(3)], axis=-1)
                tmp[roi] = image_overlay
                image_overlay = tmp

                tmp = np.ones_like(self.data['image']) * BACKGROUND_LABEL
                tmp[roi] = image_th
                image_th = tmp

            # save to data container
            self.data['image_watershed'] = image_watershed
            self.data['image_th'] = image_th
            self.data['image_overlay'] = image_overlay
            # update image view
            self._update_imageview()

    def _update_imageview(self, autoLevels=False, autoRange=False):
        txt = self.combobox_viewoptions.currentText()
        key = VIEW_OPTIONS[txt]
        try:
            self.imageView.setImage(self.data[key], autoLevels=autoLevels, autoRange=autoRange)
        except KeyError:
            pass

    @pyqtSlot(bool)
    def _toggle_roi(self, checked):
        if not checked:
            try:
                self.imageView.view.removeItem(self.data['roi'])
                self.data['roi'].sigRegionChangeFinished.disconnect(self._update_images)
                self.data['roi'] = None
            except KeyError:
                pass
        else:
            try:
                image = self.data['image']
            except KeyError:
                return

            roi = ROI(
                (0, 0),
                (image.shape[0] // 2, image.shape[1] // 2),
                movable=False, rotatable=False, resizable=False
            )
            roi.addScaleHandle([0, 1], [0.5, 0.5])
            roi.addScaleHandle([1, 0], [0.5, 0.5])
            roi.addTranslateHandle([0.5, 0.5])
            roi.sigRegionChangeFinished.connect(self._update_images)
            self.data['roi'] = roi
            self.imageView.view.addItem(roi)
        self._update_images()

    ########################
    # SETTINGS AND CLOSING #
    ########################
    def save_settings(self):
        settings = QSettings('Berto', 'InvadopodiaGui')
        settings.setValue('state', self.saveState())
        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('splitter/state', self.splitter.saveState())
        settings.setValue('splitter/geometry', self.splitter.saveGeometry())

    def read_settings(self):
        settings = QSettings('Berto', 'InvadopodiaGui')
        try:
            self.restoreState(settings.value("state"))
            self.restoreGeometry(settings.value("geometry"))
            self.splitter.restoreState(settings.value("splitter/state"))
            self.splitter.restoreGeometry(settings.value("splitter/geometry"))
        except TypeError:
            pass

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)


##############
# ALGORITHMS #
##############
def _compute_threshold_image(img, th):
    """
    Function returns the region where img < th (background region)
    :param img: array-like, shape (N, M)
    :param th: float, threshold value
    :return: array-like, region where img < th
    """
    mask = np.zeros_like(img, dtype=img.dtype)
    mask[img > th] = 255
    return mask


def _compute_watershed(img_th: np.ndarray, img_markers: np.ndarray) -> np.ndarray:
    """
    Computes the compact watershed of image, using img_th as a mask and img_markers as seed points
    :param img_th: array-like, binary image separating foreground (to be detected) and background
    :param img_markers: array-like, marker positions (x,y) used as seed for watershed algorithm
    :return: array-like, watershed of image
    """
    distance = ndi.distance_transform_edt(img_th)
    # todo: soft-code 'compactness'
    return watershed(-distance, img_markers, mask=img_th, compactness=1)


def _draw_cross(img, centre, size, colour=(1.0, 0, 0)):
    """
    Draws a cross centered at 'centre' with size 'size' directly on image 'img'
    :param img: array-like, image to modify
    :param centre: tuple, centre of cross
    :param size: float, size of cross
    :param colour: tuple, rgb values
    :return: None
    """
    assert img.ndim == 3
    # compute extrema positions
    # todo: is there a way to perform these casts in a better way?
    top = (int(centre[0]), int(centre[1] + size / 2))
    bottom = (int(centre[0]), int(centre[1] - size / 2))
    right = (int(centre[0] + size / 2), int(centre[1]))
    left = (int(centre[0] - size / 2), int(centre[1]))
    # get array slices
    rr_v, cc_v = line(*top, *bottom)
    rr_h, cc_h = line(*left, *right)
    # use color to modify the image
    img[rr_v, cc_v, :] = np.array(colour)
    img[rr_h, cc_h, :] = np.array(colour)

