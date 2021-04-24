import os.path

import numpy as np
import pandas as pd
# todo: where are we using matplotlib?
import matplotlib.pyplot as plt

from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from skimage.segmentation import watershed
from skimage.draw import line
import scipy.ndimage as ndi

from PyQt5.QtCore import QSettings, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from .Gui.mainwindow import Ui_MainWindow

import logging

VIEW_OPTIONS = {
    'Segmented image': 'image_overlay',
    'Input image': 'image',
}

mACCEPT_COLORS = [
    (0.75, 0.75, 0),
    (0, 0.75, 0),
    (0, 0.75, 0.75),
]

mIGNORE_COLORS = [
    (0.25, 0, 0)
]

# todo: threshold index and background index should be written here as program constants
# todo: markers action should be lists here as an enum or dict


class InvadopodiaGui(QMainWindow, Ui_MainWindow):
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
        # secondary view change
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
        logging.info(f'Created InvadopodiaGui - {id(self)}')

    #############################
    # CALLBACKS FOR USER ACTION #
    #############################
    def _on_load_image(self, checked):
        # todo: add file filters and customize further
        fname, _ = QFileDialog.getOpenFileName(
            None,
            'Open file',
            os.path.expanduser('~'),
            options=QFileDialog.DontUseNativeDialog
        )
        if not len(fname):
            return
        #
        try:
            image = plt.imread(fname)
            image = rgb2gray(image)
            image = np.uint8(image * 255)
        # todo: what exception to catch here ?
        except Exception as e:
            print(e)
        else:
            self.data['image'] = image
            self.data['roi'] = (slice(0, image.shape[0]-1), slice(0, image.shape[1]-1))
            # set roi values to GUI
            self.box_roi_10.setValue(image.shape[0])
            self.box_roi_11.setValue(image.shape[1])
            self.data['image_overlay'] = np.zeros_like(image)
            self.data['image_markers'] = np.zeros_like(image)
            self.data['image_watershed'] = np.zeros_like(image)
            self.data['new_marker'] = ()
            self.data['id_act_map'] = {}
            self._compute_threshold()
            self._update_imageview(autoLevels=True, autoRange=True)

    @pyqtSlot()
    def _compute_threshold(self):
        try:
            image = self.data['image']
        except KeyError:
            return
        threshold = threshold_otsu(image)
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
        pos = self.imageView.view.mapSceneToView(event.pos())
        pos = (int(pos.x()), int(pos.y()))
        print('event position: ', pos)
        # first check if the user is trying to adjust the ROI
        if self.bttn_roi_topleft.isChecked() or self.bttn_roi_bottomright.isChecked():
            self._update_roi(pos)
            event.accept()
            return

        if self.bttn_addmarker.isChecked():
            action = 'add'
        elif self.bttn_removemarker.isChecked():
            action = 'remove'
        elif self.bttn_ignoremarker.isChecked():
            action = 'ignore'
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
            index_to_action = self.data['id_act_map']
        except KeyError:
            return
        # get where user wants to save files
        fname, _ = QFileDialog.getSaveFileName(
            self,
            'Save Report',
            os.path.expanduser('~'),
        )
        if not fname:
            return
        # remove all ignored labels
        for i, action in index_to_action.items():
            if action == 'ignore':
                image_watershed[image_watershed == i] = 0

        # compute region properties
        props = regionprops_table(
            image_watershed,
            intensity_image=image,
            properties=('label', 'area', 'centroid', 'eccentricity', 'mean_intensity')
        )

        df = pd.DataFrame(props)
        scale = float(self.doublespinbox_pixeltoum.value())
        df['area'] *= scale**2

        df_summary = df.describe()
        # save to excel files
        df.to_excel(fname + '.xlsx')
        df_summary.to_excel(fname + '_summary.xlsx')

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
            index_to_action = self.data['id_act_map']
        except KeyError:
            index_to_action = {}
            self.data['id_act_map'] = index_to_action
        # check if there is a new marker from user action
        try:
            pos, action = self.data['new_marker']
            self.data['new_marker'] = None
        except (KeyError, TypeError):
            return
        # ignore markers that fall on the threshold area
        if image_th[pos] == 0:
            return

        # operate based on action
        if action == 'add' or action == 'ignore':
            index = image_markers.max() + 1
            image_markers[pos] = index
            index_to_action[index] = action
        elif action == 'remove':
            index = image_watershed[pos]
            if index != 0:
                # removes any marker present in the clicked region
                image_markers[image_watershed == index] = 0

            index_to_action.pop(index, None)
        else:
            raise RuntimeError(f'Unknown marker action {action}')
        # update the images
        self._update_images()

    def _update_images(self):
        # todo: cursor should be set to busy
        try:
            image = self.data['image']
            roi = self.data['roi']
            image_markers = self.data['image_markers']
        except KeyError:
            return

        th = float(self.box_threshold.value())
        # compute threshold overlay
        image_th = _compute_threshold_image(image, th, roi)

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
        # todo: soft-code background label
        colors = []
        try:
            index_to_action = self.data['id_act_map']
            for i, action in index_to_action.items():
                if action == 'add':
                    color = mACCEPT_COLORS[i % len(mACCEPT_COLORS)]
                else:
                    color = mIGNORE_COLORS[i % len(mIGNORE_COLORS)]
                colors.append(color)
        except KeyError:
            pass
        # add the color for threshold areas
        # todo: soft-code threshold_color
        colors.append((0, 0, 1.0))

        tmp = np.zeros_like(image_cut)
        tmp[image_th_cut == 0] = -1

        # todo: soft-code alpha, bg_label, bg_color
        image_overlay_cut = label2rgb(tmp + image_watershed_cut, image_cut, alpha=0.4, bg_label=0, bg_color=None, colors=colors)
        # restore images to the proper size
        image_watershed = np.zeros(shape=image.shape, dtype=image_watershed_cut.dtype)
        image_watershed[roi] = image_watershed_cut

        image_overlay = np.zeros(shape=(*image.shape, 3), dtype=image_overlay_cut.dtype)
        for i in range(3):
            image_overlay[:, :, i] = image
        image_overlay[roi] = image_overlay_cut

        # draw markers
        for cc in zip(*image_markers.nonzero()):
            # todo: soft-code size
            # todo: soft-code colour
            _draw_cross(image_overlay, cc, 20, (255, 0, 0))

        _draw_roi(image_overlay, roi)

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

    def _update_roi(self, pos):
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


def _compute_watershed(img_th: object, img_markers: object) -> object:
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
