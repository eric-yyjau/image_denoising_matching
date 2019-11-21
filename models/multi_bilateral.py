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
        

    def denoise(self, img, d=10, sigmaColor=0.3, sigmaSpace=1.8):
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
                LP_bilateral[i] = cv2.bilateralFilter(np.float32(LP[i]), 
                            d, sigmaColor, sigmaSpace)
            
    #        eps = 1e-4
    #        LP_bilateral = denoise_bilateral(LP-LP.min()+eps, win_size = d, 
    #                                         sigma_color = sigmaColor, 
    #                                         sigma_spatial = sigmaSpace,
    #                                         mode ='reflect',
    #                                         multichannel=True) + LP.min()
                    
#            # --- denoise HP with thresholding
            level = dcoeffs[l]
            if self.threshold_type == "BayesShrink":
                threshold = [ _bayes_thresh(channel, var) for channel in level]
            #elif self.threshold_type == "VisuShrink":
            else:
                threshold = [ _bayes_thresh(channel, var) for channel in level]
            denoised_detail = [pywt.threshold(channel, value=thres, mode=self.mode) \
                               for thres, channel in zip(threshold,level)]
#            print(LP_bilateral.shape)            
#            print(denoised_detail[0].shape,denoised_detail[1].shape,denoised_detail[2].shape)
            coeffs_rec = [LP_bilateral] + [denoised_detail]
            LP = pywt.waverec2(coeffs_rec, self.wavelet_type)
            
#            denoised_detail = [{key: pywt.threshold(level[key],
#                                            value=thresh[key],
#                                            mode=self.mode) for key in level} \
#                               for thresh, level in zip(threshold, dcoeffs)]
#                
#            coeffs_rec = [LP_bilateral] + denoised_detail
#            img_out = pywt.waverecn(coeffs_rec, self.wavelet_type)
        # channel last
        img_out = LP.transpose(1,2,0)
        
        return img_out
        
#def bilateral(img, d, sigmaColor, sigmaSpace):
#    img_out = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)    
#    return img_out
        

#%%
if __name__ == '__main__':
#    d = [50,50,50]
#    sigmaColor = [25,50,50]
#    sigmaSpace = [25,50,50]
    
    d = 50
    sigmaColor = 75
    sigmaSpace = 75
    
#    from skimage import data, img_as_float
#    from skimage.util import random_noise
#    img_gt = img_as_float(data.chelsea()[100:250, 50:300])   
#    sigma = 0.12
#    img = random_noise(img_gt, var=sigma**2)
    
    img = np.float64(cv2.imread('../data/NOISY_SRGB_010_patch.png'))/255.
    img_gt = np.float64(cv2.imread('../data/GT_SRGB_010_patch.png'))/255.
#    img_dn = bilateral_lab(img, d, sigmaColor, sigmaSpace)
    img_bl = cv2.bilateralFilter(np.float32(img), d, sigmaColor/255, sigmaSpace)
    
    multi_bilateral = MultiBilateral()
    img_dn = multi_bilateral.denoise(img)
    cv2.imshow('img',img)
    cv2.imshow('img_dn',img_dn)
    cv2.imshow('img_bl',img_bl)
    cv2.imshow('img_gt',img_gt)
    
    # B
    #cv2.imshow('img',img[:,:,0])
    