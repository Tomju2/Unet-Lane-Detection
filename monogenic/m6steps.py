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


import tensorflow as tf

from tensorflow.keras.layers import Layer


class M6(Layer):

    def __init__(self, mlt=1., s=1., wl=1., sigma=.33, **kwargs):
        super(M6, self).__init__(**kwargs)
        self.mlt = tf.Variable(mlt, name='scaling')
        self.s = tf.Variable(s, name='scale')
        self.wl = tf.Variable(wl, name='wave_length')
        self.sigma = tf.Variable(sigma, name='sigma')

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=3)
        _, cols, rows, chn = inputs.shape
        monogenic = self.monogenic_scale(cols=cols, rows=rows, s=self.s, mw=self.wl, m=self.mlt, sigma=self.sigma)
        im = tf.signal.fft3d(tf.cast(x, tf.complex64))



        imf = im * monogenic[..., 0]
        imh1 = im * monogenic[..., 1]
        imh2 = im * monogenic[..., 2]
        f = tf.math.real(tf.signal.ifft3d(imf))
        h1 = tf.math.real(tf.signal.ifft3d(imh1))
        h2 = tf.math.real(tf.signal.ifft3d(imh2))
        ori = tf.atan(tf.math.divide_no_nan(-h2, h1))
        fr = tf.sqrt(h1**2 + h2**2) + 0.001
        ft = tf.atan2(f, fr)
        ones = tf.ones_like(x)

        fts = self.scale_maxmin(ft)
        frs = self.scale_maxmin(fr)
        oris = self.scale_maxmin(ori)

        hsv_tensor_v = tf.stack((fts, frs, ones), axis=-1)
        rgb_tensor_v = self.hsv_to_rgb(hsv_tensor_v)
        hsv_tensor_o = tf.stack((oris, frs, ones), axis=-1)
        rgb_tensor_o = self.hsv_to_rgb(hsv_tensor_o)
        rgb_tensor = tf.concat([rgb_tensor_o, rgb_tensor_v, hsv_tensor_v, hsv_tensor_o], axis=-1)

        return rgb_tensor, monogenic


    def get_config(self):
        config = super().get_config().copy()
        config.update({'s': self.s, 'mw': self.wl, 'mlt': self.mlt, 'sigma': self.sigma})
        return config

    def hsv_to_rgb(self, tensor):
        h = tensor[..., 0]
        s = tensor[..., 1]
        v = tensor[..., 2]
        c = s * v
        m = v - c
        dh = h * 6
        h_category = tf.cast(dh, tf.int32)
        fmodu = dh % 2
        x = c * (1 - tf.abs(fmodu - 1))
        component_shape = tf.shape(tensor)[:-1]
        dtype = tensor.dtype
        rr = tf.zeros(component_shape, dtype=dtype)
        gg = tf.zeros(component_shape, dtype=dtype)
        bb = tf.zeros(component_shape, dtype=dtype)
        h0 = tf.equal(h_category, 0)
        rr = tf.where(h0, c, rr)
        gg = tf.where(h0, x, gg)
        h1 = tf.equal(h_category, 1)
        rr = tf.where(h1, x, rr)
        gg = tf.where(h1, c, gg)
        h2 = tf.equal(h_category, 2)
        gg = tf.where(h2, c, gg)
        bb = tf.where(h2, x, bb)
        h3 = tf.equal(h_category, 3)
        gg = tf.where(h3, x, gg)
        bb = tf.where(h3, c, bb)
        h4 = tf.equal(h_category, 4)
        rr = tf.where(h4, x, rr)
        bb = tf.where(h4, c, bb)
        h5 = tf.equal(h_category, 5)
        rr = tf.where(h5, c, rr)
        bb = tf.where(h5, x, bb)
        r = rr + m
        g = gg + m
        b = bb + m
        return tf.stack([r, g, b], axis=-1)

    def mesh_range(self, size):
        if len(size) == 1:
            rows = cols = size
        else:
            rows, cols = size
        if cols % 2:
            x_values = tf.range(-(cols-1)/2., ((cols-1)/2.)+1)/float(cols-1)
        else:
            x_values = tf.range(-cols/2., cols/2.)/float(cols)
        if rows % 2:
            y_values = tf.range(-(rows-1)/2., ((rows-1)/2.)+1)/float(rows-1)
        else:
            y_values = tf.range(-rows/2., rows/2.)/float(rows)
        return tf.meshgrid(x_values, y_values)

    def low_pass_filter(self, size, cutoff, n):
        x, y = self.mesh_range(size)
        radius = tf.sqrt(x*x + y*y)
        lpf = tf.cast(tf.signal.ifftshift(1./(1.+(radius/cutoff)**(2.*n))), tf.complex64)
        return lpf

    def meshs(self, size):
        x, y = self.mesh_range(size)
        radius = tf.cast(tf.signal.ifftshift(tf.sqrt(x*x + y*y)), tf.complex64)
        x = tf.cast(tf.signal.ifftshift(x), tf.complex64)
        y = tf.cast(tf.signal.ifftshift(y), tf.complex64)
        return  x, y, radius

    def riesz_trans(self, cols, rows):
        u1, u2, qs = self.meshs((rows,cols))
        qs = tf.cast(tf.sqrt(u1*u1 + u2*u2), tf.complex64)
        indices = tf.constant([[0, 0]])
        updates = tf.constant([1.], tf.complex64)
        q = tf.tensor_scatter_nd_update(qs, indices, updates)
        h1 = (1j * u1) / q
        h2 = (1j * u2) / q
        return h1, h2

    def scale_maxmin(self, x):
        x_min = tf.reduce_min(x, axis=(1, 2), keepdims=True)
        x_max = tf.reduce_max(x, axis=(1, 2), keepdims=True)
        scale = tf.math.divide_no_nan(
            tf.subtract(x, x_min),
            tf.subtract(x_max, x_min)
        )
        return scale

    def log_gabor_scale(self, cols, rows, s, mw, m, sigma):
        u1, u2, radf = self.meshs((rows,cols))
        indices = tf.constant([[0, 0]])
        updates = tf.constant([1.], tf.complex64)
        radius = tf.tensor_scatter_nd_update(radf, indices, updates)
        lp = self.low_pass_filter((rows, cols), .45, 15)
        log_gabor_denom = tf.cast(2. * tf.math.log(sigma)**2., tf.complex64)
        wavelength = mw * m**s
        fo = tf.constant(1., dtype=tf.float32) / wavelength
        logRadOverFo = (tf.math.log(radius / tf.cast(fo, tf.complex64)))
        log_gabor = tf.exp(-(logRadOverFo * logRadOverFo) / log_gabor_denom)
        log_gabor = (lp * log_gabor)
        return log_gabor

    def monogenic_scale(self,cols, rows, s, mw, m, sigma):
        h1, h2 = self.riesz_trans(cols, rows)
        log_gabor = self.log_gabor_scale(cols, rows, s, mw, m, sigma)
        log_gabor_h1 = log_gabor * h1
        log_gabor_h2 = log_gabor * h2
        monogenic = tf.stack([log_gabor, log_gabor_h1, log_gabor_h2], axis=-1)
        return monogenic
