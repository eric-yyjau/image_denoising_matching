# this is bilateral

"""
Created on Tue Nov 19 18:38:04 2019

OpenCV bilateral filter

@author: Yiqian
"""
import cv2
import numpy as np

def bilateral(img, d=15, sigmaColor=75/255, sigmaSpace=50):
    '''
    img - (H,W,3) or (H,W,1) intensity in [0,1]
    d: Diameter of each pixel neighborhood 
    sigmaColor: Filter sigma in the color space
    sigmaSpace: Filter sigma in the coordinate space
    '''
    img_out = cv2.bilateralFilter(np.float32(img), d, sigmaColor, sigmaSpace)    
    return img_out


#%%
def bilateral_LAB(img, d, sigmaColor, sigmaSpace):
    '''
    Denoising operates in CIE-LAB domian
    d - len 3: Diameter of each pixel neighborhood 
    sigmaColor - len 3: Filter sigma in the color space
    sigmaSpace - len 3: Filter sigma in the coordinate space
    '''
    if not isinstance(d, list):
        d = [d] * 3
    if not isinstance(sigmaColor, list):
        sigmaColor = [sigmaColor] * 3
    if not isinstance(sigmaSpace, list):
        sigmaSpace = [sigmaSpace] * 3
        
    img = np.float32(cv2.cvtColor(np.uint8(img*255), cv2.COLOR_BGR2LAB))/255.
    bilateral = []
    for i in range(img.shape[-1]):
        bilateral += [cv2.bilateralFilter(img[:,:,i], d[i], sigmaColor[i], sigmaSpace[i])]
    bilateral = np.stack(bilateral,-1)
    img_out = np.float32(cv2.cvtColor(np.uint8(bilateral*255), cv2.COLOR_LAB2BGR))/255.
    
    return img_out

#%%
def bilateral_YUV(img, d, sigmaColor, sigmaSpace):
    '''
    Denoising operates in YUV domian
    d - len 3: Diameter of each pixel neighborhood 
    sigmaColor - len 3: Filter sigma in the color space
    sigmaSpace - len 3: Filter sigma in the coordinate space
    '''
    if not isinstance(d, list):
        d = [d] * 3
    if not isinstance(sigmaColor, list):
        sigmaColor = [sigmaColor] * 3
    if not isinstance(sigmaSpace, list):
        sigmaSpace = [sigmaSpace] * 3
        
    img = np.float32(cv2.cvtColor(np.uint8(img*255), cv2.COLOR_BGR2YUV))/255.
    bilateral = []
    for i in range(img.shape[-1]):
        bilateral += [cv2.bilateralFilter(img[:,:,i], d[i], sigmaColor[i], sigmaSpace[i])]
    bilateral = np.stack(bilateral,-1)
    img_out = np.float32(cv2.cvtColor(np.uint8(bilateral*255), cv2.COLOR_YUV2BGR))/255.
    
    return img_out

#%%
if __name__ == '__main__':
    d = 50
    sigmaColor = 75/255
    sigmaSpace = 75
    
    img = np.float64(cv2.imread('../data/NOISY_SRGB_010_patch.png'))/255.
    img_gt = np.float64(cv2.imread('../data/GT_SRGB_010_patch.png'))/255.
    img_dn = bilateral_LAB(img, d, sigmaColor, sigmaSpace)
    cv2.imshow('img',img)
    cv2.imshow('img_dn',img_dn)
    cv2.imshow('img_gt',img_gt)
    