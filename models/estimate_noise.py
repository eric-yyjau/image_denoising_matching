# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:05:17 2019

2 ways to estimate noise sigma

@author: Yiqian
"""

import numpy as np
from skimage.restoration import estimate_sigma
from scipy.signal import convolve2d

def estimate_noise(img):
    '''
    Reference: D. L. Donoho and I. M. Johnstone. 
    “Ideal spatial adaptation by wavelet shrinkage.” 
    Biometrika 81.3 (1994): 425-455. DOI:10.1093/biomet/81.3.425
    '''
    return estimate_sigma(img, multichannel=True, average_sigmas=True)

def estimate_noise_fast(img):
    '''
    Reference: J. Immerkær, “Fast Noise Variance Estimation”, 
    Computer Vision and Image Understanding, 
    Vol. 64, No. 2, pp. 300-302, Sep. 1996 [PDF]
    '''
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    bgr2k = np.array([0.114, 0.587, 0.299])
    
    if len(img.shape)==3: 
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.sum(img * bgr2k, axis=-1)
        
    H,W = img.shape
    sigma = np.absolute(convolve2d(img, M)).sum()
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
    
    return sigma
        