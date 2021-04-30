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
        # ROI buttons toggle
        self.bttn_roi_topleft.toggled.connect(self._on_roibuttons_click)
        self.bttn_roi_bottomright.toggled.connect(self._on_roibuttons_click)
        # logging
        logging.debug(f'Created InvadopodiaGui - {id(self)}')

    #############################
    # CALLBACKS FOR USER ACTION #
    #############################
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
            self.data['roi'] = (slice(0, image.shape[0]-1), slice(0, image.shape[1]-1))
            # tmp_roi = ROI((0, 0), (image.shape[0]//2, image.shape[1]//2), movable=False, rotatable=False, resizable=False)
            # # todo: add a button 'set ROI' and move the ROI creating to _add_roi() method
            # ## handles scaling horizontally around center
            # tmp_roi.addScaleHandle([0, 1], [0.5, 0.5])
            # tmp_roi.addScaleHandle([1, 0], [0.5, 0.5])
            # tmp_roi.addTranslateHandle([0.5, 0.5])
            self.imageView.view.addItem(tmp_roi)
            # set roi values to GUI
            self.box_roi_10.setValue(image.shape[0])
            self.box_roi_11.setValue(image.shape[1])
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
        # first check if the user is trying to adjust the ROI
        if self.bttn_roi_topleft.isChecked() or self.bttn_roi_bottomright.isChecked():
            self._update_roi(pos)
            event.accept()
            return

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

    @pyqtSlot(bool)
    def _on_roibuttons_click(self, checked):
        if not checked:
            return
        sender = self.sender()
        if sender is self.bttn_roi_topleft:
            self.bttn_roi_bottomright.setChecked(False)
        else:
            self.bttn_roi_topleft.setChecked(False)

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
            index = image_markers.max() + 1
            image_markers[pos] = index
            label_to_action[index] = action
        elif action == mACTION_REMOVE:
            index = image_watershed[pos]
            if index != BACKGROUND_LABEL:
                # removes any marker present in the clicked region
                image_markers[image_watershed == index] = BACKGROUND_LABEL

            label_to_action.pop(index, None)
        else:
            logging.error(f'Marker at {pos} has unknown action {action}. Raising RuntimeError')
            raise RuntimeError(f'Unknown marker action {action}')
        # update the images
        self._update_images()

    def _update_images(self):
        with BusyCursor():
            try:
                image = self.data['image']
                roi = self.data['roi']
                image_markers = self.data['image_markers']
            except KeyError:
                return

            th = float(self.box_threshold.value())
            # compute threshold overlay
            image_th = _compute_threshold_image(image, th, roi)

            # todo: extend to work in case ROI is None
            # reduce image size to ROI
            image_cut = image[roi]
            image_th_cut = image_th[roi]
            image_markers_cut = image_markers[roi]

            if image_markers_cut.any():
                image_watershed_cut = _compute_watershed(image_th_cut, image_markers_cut)
            else:
                # no markers are present
                image_watershed_cut = np.zeros_like(image_cut)

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
            tmp = np.zeros_like(image_cut)
            tmp[image_th_cut == BACKGROUND_LABEL] = -1

            # todo: soft-code alpha
            image_overlay_cut = label2rgb(
                tmp + image_watershed_cut,
                image_cut,
                alpha=0.4,
                bg_label=BACKGROUND_LABEL,
                bg_color=None,
                colors=colors
            )
            # restore images to the proper size
            image_watershed = np.zeros(shape=image.shape, dtype=image_watershed_cut.dtype)
            image_watershed[roi] = image_watershed_cut

            image_overlay = np.zeros(shape=(*image.shape, 3), dtype=image_overlay_cut.dtype)
            for i in range(3):
                image_overlay[:, :, i] = image / 255
            image_overlay[roi] = image_overlay_cut

            # draw markers
            for cc in zip(*image_markers.nonzero()):
                label = image_markers[cc]
                action = label_to_action[label]
                color = mCROSS_COLORS[0] if action == mACTION_ADD else mCROSS_COLORS[1]
                # todo: soft-code size
                _draw_cross(image_overlay, cc, 20, color)

            _draw_roi(image_overlay, roi)#todo:remove

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

    def _update_roi(self, pos):#todo:remove
        roi = self.data['roi']
        if self.bttn_roi_topleft.isChecked():
            self.data['roi'] = (
                slice(pos[0], roi[0].stop),
                slice(pos[1], roi[1].stop)
            )
            # update GUI
            self.box_roi_00.setValue(pos[0])
            self.box_roi_01.setValue(pos[1])
            # uncheck button
            self.bttn_roi_topleft.setChecked(False)
        elif self.bttn_roi_bottomright.isChecked():
            self.data['roi'] = (
                slice(roi[0].start, pos[0]),
                slice(roi[1].start, pos[1])
            )
            # update GUI
            self.box_roi_10.setValue(pos[0])
            self.box_roi_11.setValue(pos[1])
            # uncheck button
            self.bttn_roi_bottomright.setChecked(False)
        else:
            return
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
def _compute_threshold_image(img, th, roi):
    """
    Function returns the region where img < th (background region)
    :param img: array-like, shape (N, M)
    :param th: float, threshold value
    :return: array-like, region where img < th
    """
    mask = np.zeros_like(img, dtype=img.dtype)
    mask[roi][img[roi] > th] = 255
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


def _draw_roi(img, roi, colour=(1.0, 0, 0)):
    """
    Draws a cross centered at 'centre' with size 'size' directly on image 'img'
    :param img: array-like, image to modify
    :param roi: tuple of slices
    :param colour: tuple, rgb values
    :return: None
    """
    assert img.ndim == 3
    # compute extrema positions
    edges = [
        (roi[0].start, roi[1].start, roi[0].stop , roi[1].start),  # |
        (roi[0].stop , roi[1].start, roi[0].stop , roi[1].stop ),  # _
        (roi[0].stop , roi[1].stop , roi[0].start, roi[1].stop ),  # |
        (roi[0].start, roi[1].stop , roi[0].start, roi[1].start),  # -
    ]
    print(edges)
    for edge in edges:
        rr, cc = line(*edge)
        img[rr, cc, :] = np.array(colour)
