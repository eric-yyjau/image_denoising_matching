# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:19:36 2019


@author: Yiqian
"""

import numpy as np
import cv2
from models.bilateral import bilateral
from models.multi_bilateral import MultiBilateral

d = 50
sigmaColor = 75
sigmaSpace = 75

img = np.float64(cv2.imread('data/NOISY_SRGB_010_patch.png'))/255.
img_gt = np.float64(cv2.imread('data/GT_SRGB_010_patch.png'))/255.
img_bl = bilateral(img, d, sigmaColor, sigmaSpace)

multi_bilateral = MultiBilateral()
img_dn = multi_bilateral.denoise(img)
cv2.imshow('img',img)
cv2.imshow('img_dn',img_dn)
cv2.imshow('img_bl',img_bl)
cv2.imshow('img_gt',img_gt)
