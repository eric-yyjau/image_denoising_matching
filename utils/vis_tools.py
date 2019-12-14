import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import logging

class Process_image(object):
    def __init__(self):
        pass


    @staticmethod
    def imgPhotometric(img, **params):
        """
        :param img:
            numpy (H, W)
        :return:
        """
        from utils.photometric import ImgAugTransform
        augmentation = ImgAugTransform(**params)
        grayscale = False
        if len(img.shape) == 2:
            grayscale = True
            img = img[:,:,np.newaxis]
        img = augmentation(img)
        # cusAug = self.customizedTransform()
        # img = cusAug(img, **self.config['augmentation'])
        if grayscale:
            img = np.squeeze(img, axis=2)
            
        return img
    # img_noise = imgPhotometric(img_np, **config['data']['augmentation'])
    # print(f"aug: {config['data']['augmentation']}")
    # plt.imshow(np.squeeze(img_noise, axis=2))
    # plt.show()

    @staticmethod
    def filter_before_matching(img, filter=None, params=None):
        """
        input: 
            img: np.float[H, W, ..] (from 0 to 1)

        """
#         from models.bilateral import bilateral, median
        from models.single_res_filters import bilateral, median
        from models.multi_bilateral import MultiBilateral
        from models.multi_guided_est import MultiGuidedEst as m_filter
        d = params['d']

        if filter is None: 
            return img
        elif filter == 'bilateral':
    #         d = 50
    #         sigmaColor = 75/255
    #         sigmaSpace = 75
            print(f"params: {params}")
            sigmaColor = params['sigmaColor']
            sigmaSpace = params['sigmaSpace']
            img = bilateral(np.float32(img), d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        elif filter == 'median':
            img = median(img, d)
            pass
        elif filter == 'm_bilateral' or filter == 'm_bilateral_thd':
            threshold_type = 'None' if filter == 'm_bilateral' else 'BayesShrink'
            multi_bilateral = MultiBilateral(wavelet_type = 'db8', wavelet_levels = 4, 
                                             threshold_type = threshold_type, #'BayesShrink', 
                                             sigma=None, mode='soft')
            sigmaColor = params['sigmaColor']
            sigmaSpace = params['sigmaSpace'] # default 1.8
            if len(img.shape) == 2:
                img = img[..., np.newaxis]
            img = multi_bilateral.denoise(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
            pass
        elif filter == 'm_guided_thd':
            the_filter = m_filter(wavelet_type = 'db8', wavelet_levels = 2, 
                threshold_type = 'BayesShrink', sigma=None, mode='soft')
            # sigmaColor = params['sigmaColor']
            sigmaSpace = params['sigmaSpace'] # default 1.8
            print(f'img: {img.shape}')
            img_rgb = np.stack([img, img, img]).transpose([1,2,0])
            img_rgb = the_filter.denoise(img_rgb, d=d, sigmaSpace=sigmaSpace)
            img = img_rgb[...,0]
            pass
        else:
            logging.warning(f"filter type {filter} not defined")
    
        
        return img

    @staticmethod
    def update_config(config, mode=1, param=0, if_print=True):
        if mode == 0:
            pass
            # config['training']['pretrained'] = pretrained
            # config['training']['pretrained_SP'] = pretrained_SP
        elif mode == 1:
            config['data']['augmentation']['photometric']['enable'] = True
            assert config['data']['augmentation']['photometric']['enable'] == True
            config['data']['augmentation']['photometric']['params']['additive_gaussian_noise']['stddev_range'] = param
        elif mode == 2:
            config['data']['augmentation']['photometric']['enable'] = True
            assert config['data']['augmentation']['photometric']['enable'] == True
            config['data']['augmentation']['photometric']['params']['additive_gaussian_noise']['stddev_range'] = param
            config['model']['filter'] = 'bilateral'

        if if_print and mode <= 5:
            # logging.info(f"update params: {config['data']['augmentation']}")
            print(f"update params: {config['data']['augmentation']}")
        files_list = []

        return config, files_list        

    @staticmethod
    def get_bilateral_params(sigma, d=11):
    #     sigma_n = 25/255
        sigma_n = sigma/255
    #     bilateral_params = {
    # #         'd': 0,
    #         'd': 25,
    #         'sigmaColor': sigma_n*2,
    # #         'sigmaColor': 75/255,
    #         'sigmaSpace': 75,
    # #         'sigmaSpace': 3*sigma,
    #     }

        bilateral_params = {
    #         'd': 0,
            'd': d,
            'sigmaColor': sigma_n*2,
    #         'sigmaColor': 75/255,
            'sigmaSpace': 1.8,
    #         'sigmaSpace': 3*sigma,
        }
        return bilateral_params

# sigma=0
# bilateral_params = {
#     'd': 0,
#     'sigmaColor': 75/255,
#     'sigmaSpace': 2*sigma,
# }

# img_np = img.numpy().squeeze()
# print(f"{img_np.shape}")
# img_bl = filter_before_matching(img_np, filter='bilateral', params=bilateral_params)
# print(f"img: {img_bl.shape}")
# plt.imshow(img_np, cmap='gray')
# plt.show()
# plt.imshow(img_bl, cmap='gray')
# plt.show()

