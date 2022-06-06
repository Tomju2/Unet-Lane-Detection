#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["E. Ulises Moya", "Sabastia Xambo", "Abraham Sanchez", "Raul Nanclares", "Jorge Martinez", "Alexander Quevedo", "Baowei Fei"]
__copyright__ = "Copyright 2020, Gobierno de Jalisco, Universitat Politecnica de Catalunya, University of Texas at Dallas"
__credits__ = ["E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["E. Ulises Moya", "Abraham Sanchez"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from monogenic.m6steps import M6


def apply_monogenic(img, size):
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

    #input image --- step 1
    #plt.imshow(img)
    #plt.title(''), plt.xticks([]), plt.yticks([])
    #plt.show()


    #-Resize the image
    img = np.expand_dims(img, axis=0)
    #print(img.shape)


    #-normalization and float
    img = img.astype(np.float32)
    img = img / 255.
    #- change contrast

    #img = change_contrast(img, .9)


    #--------------------------------------M6Layer
    #-m6layer paramters
    m6 = M6(s=3., wl=10., mlt=1.2, sigma=0.35)
    #-m6 computation
    m6ly, monogenic = m6(img)


    m6ly = tf.reshape(m6ly, [size, size, 12])
    m6ly = tf.cast(m6ly, tf.float32)

    out= m6ly

    #-----------output of M6

    phasergb=out[:, :, 3:6]
    orirgb=out[:, :, :3]
    phasehsv=out[:, :, 9:12]
    orihsv=out[:, :, 6:9]

    orihsv = cv2.cvtColor(orihsv.numpy(), cv2.COLOR_BGR2GRAY)
    orihsv = np.expand_dims(orihsv, axis=2)

    return orihsv
