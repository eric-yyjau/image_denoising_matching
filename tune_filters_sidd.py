# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:31:02 2019

@author: No Name
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
random_per = np.random.permutation(len(foldernames)).tolist()

#%%
#random_per = np.random.permutation(len(foldernames)).tolist()

for wavelet_levels in [1,2,3,4]:
    for wavelet_type in ['haar', 'bior3.1', 'bior3.3', 'bior3.5', 'db2', 'db4', 'db8', 'sym2' , 'sym4', 'sym8']:
        print(wavelet_levels, wavelet_type)
        multi_bilateral = MultiBilateralEst(wavelet_type = wavelet_type, wavelet_levels = wavelet_levels, 
                         threshold_type = 'BayesShrink', sigma=None, mode='soft')
        
        ''' Test in dataset '''
        PSNR, SSIM, MSE = [],[],[]
        
        for idx, name in enumerate([foldernames[i] for i in random_per[:40]]):
            img = np.float64(cv2.imread(os.path.join(path,name,'NOISY_SRGB_010.PNG')))/255.
            img_gt = np.float64(cv2.imread(os.path.join(path,name,'GT_SRGB_010.PNG')))/255. 
            
            img_dn = multi_bilateral.denoise(img, d=11, sigmaSpace=1.8)
            
            psnr, ssim, mse = denoise_eval(img_gt, img_dn, print_result =False)
            
            PSNR.append(psnr)
            SSIM.append(ssim)
            MSE.append(mse)
            
#            if idx % 10 == 0:
#                print(idx)
        
        print('After denoising:')
        print('PSNR: %.2f' %  np.array(PSNR).mean())
        print('SSIM: %.4f' %  np.array(SSIM).mean())
        print('MSE: %.2f' %  np.array(MSE).mean())