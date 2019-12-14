# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:01:59 2019

multiresolution bilateral filtering

@author: Yiqian
"""

import cv2
import numpy as np
import pywt
import scipy

from skimage.restoration import (denoise_bilateral, estimate_sigma)
from scipy.signal import convolve2d 

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

def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details*details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh

def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.
    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.
    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).
    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma

#%%
class MultiBilateralEst():
    """
    sigmaColor is estimated at each level
    Only need to specify window size (d) and sigmaSpace
    
    Returns
    -------
    out : ndarray
        Denoised image.
    """
    
    def __init__(self, wavelet_type = 'db8', wavelet_levels = 4, 
                 threshold_type = 'BayesShrink', sigma=None, mode='soft'):
        
        self.wavelet_type = wavelet_type
        self.wavelet_levels = wavelet_levels
        self.threshold_type = threshold_type
        self.sigma = sigma
        self.mode = mode
        

    def denoise(self, img, d=11, sigmaColor = 50/255, sigmaSpace=1.8):
        # channel first
        img = img.transpose(2,0,1)
        # --- wavelet transform
        coeffs = pywt.wavedec2(img, wavelet = self.wavelet_type, 
                               level = self.wavelet_levels)
        LP = coeffs[0]
        dcoeffs = coeffs[1:]
        
        if self.sigma is None:
            # Estimate the noise via the method in [2]_
            detail_coeffs = dcoeffs[-1][-1]
            self.sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
        var = self.sigma**2
        
        for l in range(self.wavelet_levels):
            # --- denoise LP with bilateral
            LP_bilateral = np.zeros_like(LP)
            for i in range(LP.shape[0]):
                s = estimate_noise_fast(LP[i])
                LP_bilateral[i] = cv2.bilateralFilter(np.float32(LP[i]), 
                            d, s*2, sigmaSpace)
                                
#            # --- denoise HP with thresholding
            level = dcoeffs[l]
            if self.threshold_type == "BayesShrink":
                threshold = [ _bayes_thresh(channel, var) for channel in level]
                denoised_detail = [pywt.threshold(channel, value=thres, mode=self.mode) \
                                   for thres, channel in zip(threshold,level)]
            # TODO: add VisuShrink
            #elif self.threshold_type == "VisuShrink": 
            else:
                denoised_detail = level
            coeffs_rec = [LP_bilateral] + [denoised_detail]
            LP = pywt.waverec2(coeffs_rec, self.wavelet_type)
            
        # channel last
        img_out = LP.transpose(1,2,0)
        if sigmaColor is None:
            sigmaColor = 2*estimate_noise_fast(img_out)
        img_out = cv2.bilateralFilter(np.float32(img_out),
                                      d, sigmaColor, sigmaSpace)
        
        return img_out
                

#%%
if __name__ == '__main__':
#    d = [50,50,50]
#    sigmaColor = [25,50,50]
#    sigmaSpace = [25,50,50]
    
    d = 50
    sigmaSpace = 75
    
#    from skimage import data, img_as_float
#    from skimage.util import random_noise
#    img_gt = img_as_float(data.chelsea()[100:250, 50:300])   
#    sigma = 0.12
#    img = random_noise(img_gt, var=sigma**2)
    
    img = np.float64(cv2.imread('../data/NOISY_SRGB_010_patch.png'))/255.
    img_gt = np.float64(cv2.imread('../data/GT_SRGB_010_patch.png'))/255.
#    img_dn = bilateral_lab(img, d, sigmaColor, sigmaSpace)
    
    multi_bilateral = MultiBilateralEst()
    img_dn = multi_bilateral.denoise(img)
    cv2.imshow('img',img)
    cv2.imshow('img_dn',img_dn)
    cv2.imshow('img_gt',img_gt)
    
    # B
    #cv2.imshow('img',img[:,:,0])
    