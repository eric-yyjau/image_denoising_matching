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
class MultiBilateral():
    """Perform wavelet thresholding.
    Parameters
    ----------
    image : ndarray (2d or 3d) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    wavelet : string
        The type of wavelet to perform. Can be any of the options
        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,
        db3, db4, haar}``.
    threshold_type : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. 
    sigma : float, optional
        The standard deviation of the noise. The noise is estimated when sigma
        is None (the default) by the method in [2]_.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels
        (see Notes below).
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
        

    def denoise(self, img, d=50, sigmaColor=0.04, sigmaSpace=0.04):
        # --- wavelet transform
        original_extent = tuple(slice(s) for s in img.shape)
        coeffs = pywt.wavedecn(img, wavelet = self.wavelet_type, 
                               level = self.wavelet_levels)
        LP = coeffs[0]
        dcoeffs = coeffs[1:]
        
        # --- denoise LP with bilateral
#        LP_bilateral = LP
#        for i in range(LP.shape[-1]):
#            LP_bilateral[:,:,i] = cv2.bilateralFilter(np.float32(LP[:,:,i]), 
#                        d, sigmaColor, sigmaSpace)
        eps = 1e-4
        LP_bilateral = denoise_bilateral(LP - LP.min()+eps, d, sigmaColor, sigmaSpace,
                                         multichannel=True) + LP.min()
                
        # --- denoise HP with thresholding
        if self.sigma is None:
            # Estimate the noise via the method in [2]_
            detail_coeffs = dcoeffs[-1]['d' * img.ndim]
            self.sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
            
        var = self.sigma**2
        print(self.sigma)
        
        if self.threshold_type == "BayesShrink":
            threshold = [{key: _bayes_thresh(level[key], var) \
                          for key in level} for level in dcoeffs]
        #elif self.threshold_type == "VisuShrink":
        else: # use BayesShrink
            threshold = [{key: _bayes_thresh(level[key], var) \
                          for key in level} for level in dcoeffs]
        print(threshold)
        denoised_detail = [{key: pywt.threshold(level[key],
                                        value=thresh[key],
                                        mode=self.mode) for key in level} \
                           for thresh, level in zip(threshold, dcoeffs)]
            
        coeffs_rec = [LP_bilateral] + denoised_detail
        img_out = pywt.waverecn(coeffs_rec, self.wavelet_type)[original_extent]
        
        return img_out
        
#def bilateral(img, d, sigmaColor, sigmaSpace):
#    img_out = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)    
#    return img_out
        
#%%
def bilateral_lab(img, d, sigmaColor, sigmaSpace):
    '''
    Denoising operates in CIE-LAB domian
    d - len 3: Diameter of each pixel neighborhood 
    sigmaColor - len 3: Filter sigma in the color space
    sigmaSpace - len 3: Filter sigma in the coordinate space
    '''
        
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bilateral_l = cv2.bilateralFilter(img_lab[:,:,0], d[0], sigmaColor[0], sigmaSpace[0])
    bilateral_a = cv2.bilateralFilter(img_lab[:,:,1], d[1], sigmaColor[1], sigmaSpace[1])
    bilateral_b = cv2.bilateralFilter(img_lab[:,:,2], d[2], sigmaColor[2], sigmaSpace[2])
    bilateral = np.stack((bilateral_l,bilateral_a,bilateral_b),-1)
    img_out = cv2.cvtColor(bilateral, cv2.COLOR_LAB2BGR)
    
    return img_out


#%%
if __name__ == '__main__':
#    d = [50,50,50]
#    sigmaColor = [25,50,50]
#    sigmaSpace = [25,50,50]
    
#    d = 50
#    sigmaColor = 75
#    sigmaSpace = 75
    
    img = np.float64(cv2.imread('../data/NOISY_SRGB_010_patch.png'))/255.
    img_gt = np.float64(cv2.imread('../data/GT_SRGB_010_patch.png'))/255.
#    img_dn = bilateral_lab(img, d, sigmaColor, sigmaSpace)
#    img_dn = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    
    multi_bilateral = MultiBilateral()
    img_dn = multi_bilateral.denoise(img)
    cv2.imshow('img_dn',img_dn)
    cv2.imshow('img_gt',img_gt)
    cv2.imshow('img',img)
    
    # B
    #cv2.imshow('img',img[:,:,0])