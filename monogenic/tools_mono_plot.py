
import tensorflow as tf
import numpy as np
from scipy.fftpack import ifftshift
import cv2

def filtergrid(rows, cols):

    # Set up u1 and u2 matrices with ranges normalised to +/- 0.5
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)

    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)

    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2)

    return radius, u1, u2


def lowpassfilter(size, cutoff, n):
    """
    Constructs a low-pass Butterworth filter:

        f = 1 / (1 + (w/cutoff)^2n)

    usage:  f = lowpassfilter(sze, cutoff, n)

    where:  size    is a tuple specifying the size of filter to construct
            [rows cols].
        cutoff  is the cutoff frequency of the filter 0 - 0.5
        n   is the order of the filter, the higher n is the sharper
            the transition is. (n must be an integer >= 1). Note
            that n is doubled so that it is always an even integer.

    The frequency origin of the returned filter is at the corners.
    """

    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))



def circle_input(size, cutoff, n):

	lp = lowpassfilter(size, cutoff, n)
	circle = np.fft.fftshift(lp)

	return circle	


def line_input(size, lenth, theta):
	rows, cols = size #have to be even and equal
	#lenth #have to be odd
	img = np.zeros([rows,cols], dtype=np.uint8)
	r4= rows/4
	ry0 = r4
	ry1 = 3*r4
	rx0 = 2*r4 - lenth/2
	rx1 = 2*r4 + lenth/2
	img[int(ry0):int(ry1),int(rx0):int(rx1)]= 255
	M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
	im = cv2.warpAffine(img,M,(cols,rows))	
	return im
		


def change_contrast(image, level):
    image = tf.image.adjust_contrast(image, level)
    return image 

def change_brightness(image, level):
    image = tf.image.adjust_brightness(image,level)
    return image

