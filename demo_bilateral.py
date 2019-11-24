# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:19:36 2019


@author: Yiqian
"""

import numpy as np
import cv2
from models.single_res_filters import bilateral, median, guided
from models.multi_bilateral import MultiBilateral
from evaluations.denoise_eval import denoise_eval

#%%
d = 11
sigmaColor = 50/255
sigmaSpace = 50

img = np.float64(cv2.imread('data/NOISY_SRGB_081.png'))/255.
img_gt = np.float64(cv2.imread('data/GT_SRGB_081.png'))/255.
#img_bl = bilateral(img, 11, sigmaColor, 1.8)
img_bl2 = bilateral(img, d, sigmaColor, sigmaSpace)
img_md = median(img, d)
multi_bilateral = MultiBilateral(wavelet_type = 'db8', wavelet_levels = 4, 
                                 threshold_type = 'BayesShrink', 
                                 sigma=None, mode='soft')
img_dn = multi_bilateral.denoise(img, d=d, sigmaColor=sigmaColor, sigmaSpace=1.8)

cv2.imshow('img',img[:256,:256])
cv2.imshow('img_md',img_md)
cv2.imshow('img_bl2',img_bl2)
cv2.imshow('img_dn',img_dn)
cv2.imshow('img_gt',img_gt)

cv2.imwrite('img_big.png',np.uint8(img*255))
cv2.imwrite('img_big_md.png',np.uint8(img_md*255))
cv2.imwrite('img_big_bl.png',np.uint8(img_bl2*255))
cv2.imwrite('img_big_dn.png',np.uint8(img_dn*255))
cv2.imwrite('img_big_gt.png',np.uint8(img_gt*255))

print('before:')
p,s,m = denoise_eval(img_gt,img, print_result =True)
print('median:')
p,s,m = denoise_eval(img_gt,img_md, print_result =True)
print('bilateral:')
p,s,m = denoise_eval(img_gt,img_bl2, print_result =True)
print('multi-bilateral:')
p,s,m = denoise_eval(img_gt,img_dn, print_result =True)