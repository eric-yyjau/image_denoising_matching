# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:19:36 2019


@author: Yiqian
"""

import numpy as np
import cv2
from models.single_res_filters import bilateral, median, guided
from models.multi_bilateral import MultiBilateral
from models.multi_bilateral_est import MultiBilateralEst
from models.multi_guided_est import MultiGuidedEst
from evaluations.denoise_eval import denoise_eval

from skimage.restoration import (denoise_wavelet, estimate_sigma)

#%%
d = 11
sigmaColor = 50/255
sigmaSpace = 50

#img = np.float64(cv2.imread('data/img.png'))/255.
#img_gt = np.float64(cv2.imread('data/img_gt.png'))/255.
img = np.float64(cv2.imread('data/NOISY_SRGB_010_patch.png'))/255.
img_gt = np.float64(cv2.imread('data/GT_SRGB_010_patch.png'))/255.


''' Define filters '''
multi_bilateral = MultiBilateralEst(wavelet_type = 'db4', wavelet_levels = 2, 
                 threshold_type = 'BayesShrink', sigma=None, mode='soft')
multi_bilateral_nothres = MultiBilateralEst(wavelet_type = 'db4', wavelet_levels = 2, 
                 threshold_type = 'None', sigma=None, mode='soft')
multi_guided = MultiGuidedEst(wavelet_type = 'db4', wavelet_levels = 2, 
                 threshold_type = 'BayesShrink', sigma=None, mode='soft')
#
img_gd = guided(img,img, 3, 1000)
img_bl = bilateral(img, 11, 100/255, 1.8)
img_md = median(img, d=11)
img_mbt = np.clip(multi_bilateral.denoise(img, d=11, sigmaSpace=1.8),0,1)
img_mb = np.clip(multi_bilateral_nothres.denoise(img, d=11, sigmaSpace=1.8),0,1)
img_wt = denoise_wavelet(img, multichannel=True, convert2ycbcr=False,
                       method='BayesShrink', mode='soft')
img_mgt = np.clip(multi_guided.denoise(img, d=3),0,1)



#cv2.imwrite('img_wt.png',np.uint8(img_wt*255))
cv2.imwrite('img_md.png',np.uint8(img_md*255))
#cv2.imwrite('img_bl.png',np.uint8(img_bl*255))
#cv2.imwrite('img_gd.png',np.uint8(img_gd*255))
#cv2.imwrite('img_mb.png',np.uint8(img_mb*255))
#cv2.imwrite('img_mbt.png',np.uint8(img_mbt*255))
#cv2.imwrite('img_mgt.png',np.uint8(img_mgt*255))
#cv2.imshow('img_mgt',img_mgt)

#
print('before:')
p,s,m = denoise_eval(img_gt,img, print_result =True)
print('wavelet_thres:')
p,s,m = denoise_eval(img_gt,img_wt, print_result =True)
print('median:')
p,s,m = denoise_eval(img_gt,img_md, print_result =True)
print('bilateral:')
p,s,m = denoise_eval(img_gt,img_bl, print_result =True)
print('guided:')
p,s,m = denoise_eval(img_gt,img_gd, print_result =True)
print('multi-bilateral:')
p,s,m = denoise_eval(img_gt,img_mb, print_result =True)
print('multi-bilateral_t:')
p,s,m = denoise_eval(img_gt,img_mbt, print_result =True)
print('multi-guided_t:')
p,s,m = denoise_eval(img_gt,img_mgt, print_result =True)