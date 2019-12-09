# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:16:34 2019

@author: No Name
"""

import numpy as np
import cv2
#from models.smooth import box_filter

def box_filter(I,r):
#    return cv2.blur(I,(r,r),borderType=cv2.BORDER_REFLECT)
    return cv2.blur(I,(r,r))

def to_32F(image):
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(np.float32(image), 0, 1)


def to_8U(image):
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(np.uint8(image), 0, 255)


class GuidedFilter():
    """
    This is a factory class which builds guided filter
    according to the channel number of guided Input.
    The guided input could be gray image, color image,
    or multi-dimensional feature map.
    References:
        K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    def __init__(self, I, radius, eps):
        """
        Parameters
        ----------
        I: NDArray
            Guided image or guided feature map
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        if len(I.shape) == 2:
            self._Filter = GrayGuidedFilter(I, radius, eps)
        else:
            self._Filter = MultiDimGuidedFilter(I, radius, eps)

    def filter(self, p):
        """
        Parameters
        ----------
        p: NDArray
            Filtering input which is 2D or 3D with format
            HW or HWC
        Returns
        -------
        ret: NDArray
            Filtering output whose shape is same with input
        """
        p = to_32F(p)
        if len(p.shape) == 2:
            return self._Filter.filter(p)
        elif len(p.shape) == 3:
            channels = p.shape[2]
            ret = np.zeros_like(p, dtype=np.float32)
            for c in range(channels):
                ret[:, :, c] = self._Filter.filter(p[:, :, c])
            return ret


class GrayGuidedFilter():
    """
    Specific guided filter for gray guided image.
    """
    def __init__(self, I, radius, eps):
        """
        Parameters
        ----------
        I: NDArray
            2D guided image
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        self.I = to_32F(I)
        self.radius = radius
        self.eps = eps

    def filter(self, p):
        """
        Parameters
        ----------
        p: NDArray
            Filtering input of 2D
        Returns
        -------
        q: NDArray
            Filtering output of 2D
        """
        # step 1
        meanI  = box_filter(I=self.I, r=self.radius)
        meanp  = box_filter(I=p, r=self.radius)
        corrI  = box_filter(I=self.I * self.I, r=self.radius)
        corrIp = box_filter(I=self.I * p, r=self.radius)
        # step 2
        varI   = corrI - meanI * meanI
        covIp  = corrIp - meanI * meanp
        # step 3
        a      = covIp / (varI + self.eps)
        b      = meanp - a * meanI
        # step 4
        meana  = box_filter(I=a, r=self.radius)
        meanb  = box_filter(I=b, r=self.radius)
        # step 5
        q = meana * self.I + meanb

        return q


class MultiDimGuidedFilter():
    """
    Specific guided filter for color guided image
    or multi-dimensional feature map.
    """
    def __init__(self, I, radius, eps):
        self.I = to_32F(I)
        self.radius = radius
        self.eps = eps

        self.rows = self.I.shape[0]
        self.cols = self.I.shape[1]
        self.chs  = self.I.shape[2]

    def filter(self, p):
        """
        Parameters
        ----------
        p: NDArray
            Filtering input of 2D
        Returns
        -------
        q: NDArray
            Filtering output of 2D
        """
#        p_ = np.expand_dims(p, axis=2)
        p_ = p

        meanI = box_filter(I=self.I, r=self.radius) # (H, W, C)
        meanp = box_filter(I=p_, r=self.radius) # (H, W, 1)
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)
        meanI_ = meanI.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        corrI_ = np.matmul(I_, I_.transpose(0, 2, 1))  # (HW, C, C)
        corrI_ = corrI_.reshape((self.rows, self.cols, self.chs*self.chs)) # (H, W, CC)
        corrI_ = box_filter(I=corrI_, r=self.radius)
        corrI = corrI_.reshape((self.rows*self.cols, self.chs, self.chs)) # (HW, C, C)

        U = np.expand_dims(np.eye(self.chs, dtype=np.float32), axis=0)
        # U = np.tile(U, (self.rows*self.cols, 1, 1)) # (HW, C, C)

        left = np.linalg.inv(corrI + self.eps * U) # (HW, C, C)

        corrIp = box_filter(I=self.I*p_, r=self.radius) # (H, W, C)
        covIp = corrIp - meanI * meanp # (H, W, C)
        right = covIp.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        a = np.matmul(left, right) # (HW, C, 1)
        axmeanI = np.matmul(a.transpose((0, 2, 1)), meanI_) # (HW, 1, 1)
        axmeanI = axmeanI.reshape((self.rows, self.cols, 1))
        b = meanp - axmeanI # (H, W, 1)
        a = a.reshape((self.rows, self.cols, self.chs))

        meana = box_filter(I=a, r=self.radius)
        meanb = box_filter(I=b, r=self.radius)

        meana = meana.reshape((self.rows*self.cols, 1, self.chs))
        meanb = meanb.reshape((self.rows*self.cols, 1, self.chs))
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1))

        q = np.matmul(meana, I_) + meanb
        q = q.reshape((self.rows, self.cols,self.chs))

        return q