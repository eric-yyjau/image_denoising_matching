# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:40:04 2019

PSNR, SSIM, MSE

@author: Yiqian
"""

import cv2
import numpy as np
from skimage.measure import compare_ssim

def PSNR(img1, img2):
    ''' 
    input: img intensity between [0,1] 
    output: psnr between [0,1], the higher the better
    '''
    return cv2.PSNR(np.uint8(img1*255),np.uint8(img2*255))

def PSNR_self(img1, img2):
    ''' 
    input: img intensity between [0,1] 
    output: psnr between [0,1], the higher the better
    '''
    mse = ( (np.uint8(img1*255) - np.uint8(img2*255)) ** 2 ).mean()
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def SSIM(img1, img2):
    ''' 
    input: img intensity between [0,1] 
    output: ssim between [0,1], the higher the better
    '''
    img1 = cv2.cvtColor(np.uint8(img1*255), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(np.uint8(img2*255), cv2.COLOR_BGR2GRAY)
    return compare_ssim(img1,img2)

def MSE(img1, img2):
    ''' 
    input: img intensity between [0,1] 
    output: mse between [0, 255**2], the lower the better
    '''
    img1 = np.uint8(img1*255)
    img2 = np.uint8(img2*255)
    return ((img1-img2)**2).mean()

def denoise_eval(img1, img2, print_result = False):
#    psnr = PSNR(img1, img2)
    psnr = PSNR_self(img1, img2)
    ssim = SSIM(img1, img2)
    mse = MSE(img1, img2)
    if print_result:
        print('PSNR: %.4f' % psnr)
        print('SSIM: %.4f' % ssim)
        print('MSE: %.4f' % mse)
    return psnr, ssim, mse