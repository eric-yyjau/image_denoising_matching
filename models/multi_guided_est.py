# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:39:22 2019

@author: No Name
"""

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

from models.single_res_filters import guided
from models.guided import GuidedFilter, GrayGuidedFilter, MultiDimGuidedFilter

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
def flt2norm(img_flt):
    imax = img_flt.max()
    imin = img_flt.min()
    img_norm = (img_flt-imin)/(imax-imin)
    return img_norm, imin, imax


def norm2flt(img_norm, imin, imax):
    return img_norm*(imax-imin) + imin

#%%
class MultiGuidedEst():
    """
    sigmaColor is estimated at each level
    Only need to specify window size (d) and sigmaSpace
    
    Returns
    -------
    out : ndarray
        Denoised image.
    """
    
    def __init__(self, wavelet_type = 'db8', wavelet_levels = 2, 
                 threshold_type = 'BayesShrink', sigma=None, mode='soft'):
        
        self.wavelet_type = wavelet_type
        self.wavelet_levels = wavelet_levels
        self.threshold_type = threshold_type
        self.sigma = sigma
        self.mode = mode
        

    def denoise(self, img, d=3):
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
            LP_filter = np.zeros_like(LP)
            for i in range(LP.shape[0]):
                s = 2*estimate_noise_fast(LP[i])
#                GF = GrayGuidedFilter(LP[i], radius=d, eps=1e-6)
#                GF = GrayGuidedFilter(LP[i], radius=d, eps=1e-4)                
#                LP_filter[i] = GF.filter(LP[i])
                im, imin, imax = flt2norm(LP[i])
                im_out = guided(im, im, d=d, sigmaColor=(s**2)*255*255, color_space = 'RGB')
#                im_out = guided(im, im, d=d, sigmaColor=1e-6*255*255, color_space = 'RGB')
                LP_filter[i] = norm2flt(im_out, imin, imax)
                                
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
            coeffs_rec = [LP_filter] + [denoised_detail]
            LP = pywt.waverec2(coeffs_rec, self.wavelet_type)
            
        # channel last
        img_out = LP.transpose(1,2,0)
#        sigmaColor = estimate_noise_fast(img_out)
        im, imin, imax = flt2norm(img_out)
        im_out = guided(im, im, d=d, sigmaColor=2500, color_space = 'RGB')
#        im_out = guided(im, im, d=d, sigmaColor=1e-5*255*255, color_space = 'RGB')
        img_out = norm2flt(im_out, imin, imax)
        
#        GF = MultiDimGuidedFilter(img_out, radius=d, eps=1e-5) 
#        GF = MultiDimGuidedFilter(img_out, radius=d, eps=2500/255/255)                
#        img_out = GF.filter(img_out)
        
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
    
    img = np.float64(cv2.imread('../img_big.png'))/255.
    img_gt = np.float64(cv2.imread('../img_big_gt.png'))/255.
#    img = np.float64(cv2.imread('../data/NOISY_SRGB_010_patch.png'))/255.
#    img_gt = np.float64(cv2.imread('../data/GT_SRGB_010_patch.png'))/255.
#    img_dn = bilateral_lab(img, d, sigmaColor, sigmaSpace)
    
    multi_guided = MultiGuidedEst()
    img_dn = multi_guided.denoise(img)
    cv2.imshow('img',img)
    cv2.imshow('img_dn',img_dn)
    cv2.imshow('img_gt',img_gt)
    
    cv2.imwrite('img_gd.png',img_dn)
    # B
    #cv2.imshow('img',img[:,:,0])
    