# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:55:12 2019



@author: Yiqian
"""

import numpy as np
import cv2
import os
from models.single_res_filters import bilateral, median, guided
from models.multi_bilateral import MultiBilateral
from models.multi_bilateral_est import MultiBilateralEst
from models.multi_guided_est import MultiGuidedEst
from evaluations.denoise_eval import denoise_eval

from skimage.restoration import (denoise_wavelet, estimate_sigma)

path = "../SIDD_Small_sRGB_Only/Data"
walk = [x for x in os.walk(path)]
foldernames = walk[0][1]

#%%
#img = np.float64(cv2.imread('data/NOISY_SRGB_010_patch.png'))/255.
#img_gt = np.float64(cv2.imread('data/GT_SRGB_010_patch.png'))/255.

''' Define filters '''
multi_bilateral = MultiBilateralEst(wavelet_type = 'db8', wavelet_levels = 3, 
                 threshold_type = 'BayesShrink', sigma=None, mode='soft')
#multi_bilateral_nothres = MultiBilateralEst(wavelet_type = 'db8', wavelet_levels = 2, 
#                 threshold_type = 'None', sigma=None, mode='soft')
#multi_guided = MultiGuidedEst(wavelet_type = 'db8', wavelet_levels = 2, 
#                 threshold_type = 'BayesShrink', sigma=None, mode='soft')

#img_dn = multi_bilateral.denoise(img)

#img_dn = {}
''' Test in dataset '''
#PSNR_before, SSIM_before, MSE_before = 0, 0, 0
PSNR, SSIM, MSE = [],[],[]

#random_per = np.random.permutation(len(foldernames)).tolist()
random_per = np.random.permutation(len(foldernames))[:40].tolist()
for idx, name in enumerate([foldernames[i] for i in random_per]):
    img = np.float64(cv2.imread(os.path.join(path,name,'NOISY_SRGB_010.PNG')))/255.
    img_gt = np.float64(cv2.imread(os.path.join(path,name,'GT_SRGB_010.PNG')))/255. 
    
#    img_dn = guided(img,img, 25, 2500)
#    img_dn = bilateral(img, 25, 50/255, 1.8)
#    img_dn = median(img, d=25)
    img_dn = multi_bilateral.denoise(img, d=11, sigmaSpace=1.8)
#    img_dn = multi_bilateral_nothres.denoise(img, d=11, sigmaSpace=1.8)
#    img_dn = denoise_wavelet(img, multichannel=True, convert2ycbcr=False,
#                           method='BayesShrink', mode='soft')
#    img_dn = multi_guided.denoise(img, d=3, sigmaSpace=1.8)
    
#    p, s, m = denoise_eval(img_gt, img, print_result =False)
    psnr, ssim, mse = denoise_eval(img_gt, img_dn, print_result =False)
    
#    PSNR_before += p
#    SSIM_before += s
#    MSE_before += m
    PSNR.append(psnr)
    SSIM.append(ssim)
    MSE.append(mse)
    
    if idx % 10 == 0:
        print(idx)

#PSNR_before = PSNR_before / len(foldernames)
#SSIM_before = SSIM_before / len(foldernames)
#MSE_before = MSE_before / len(foldernames)
        
#print('Before denoising:')
#print('PSNR', PSNR_before)
#print('SSIM', SSIM_before)
#print('MSE', MSE_before)

print('After denoising:')
print('PSNR: %.2f' %  np.array(PSNR).mean())
print('SSIM: %.4f' %  np.array(SSIM).mean())
print('MSE: %.2f' %  np.array(MSE).mean())

#    img_dn['multi_median'] = multi_median.denoise(img)
#    img_dn['multi_bilateral'] = multi_bilateral.denoise(img)
#    img_dn['multi_bilateral_nothres'] = multi_bilateral_nothres.denoise(img)
    
#cv2.imshow('img',img)
#cv2.imshow('img_dn',img_dn)
#cv2.imshow('img_gt',img_gt)
