# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:11:57 2019

@author: No Name
"""


import numpy as np
import cv2
import pywt

img = np.float64(cv2.imread('data/NOISY_SRGB_010_patch.png'))/255.
img_gt = np.float64(cv2.imread('data/GT_SRGB_010_patch.png'))/255.

#%%
L = 1
img_coeffs = pywt.wavedec2(img.transpose(2,0,1), wavelet = 'db4', level = L)
gt_coeffs = pywt.wavedec2(img_gt.transpose(2,0,1), wavelet = 'db4', level = L)
noise_LP1 = img_coeffs[0]-gt_coeffs[0]
cv2.imwrite('noise_LP'+str(L)+'.png',
            np.uint8((noise_LP1-noise_LP1.min())/ \
                     (noise_LP1.max()-noise_LP1.min())*255).transpose(1,2,0))

cv2.imwrite('img_LP'+str(L)+'.png',np.uint8((img_coeffs[0]-img_coeffs[0].min())/ \
                     (img_coeffs[0].max()-img_coeffs[0].min())*255).transpose(1,2,0))