import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

# from .base_dataset import BaseDataset
# from .utils import pipeline
from utils.tools import dict_update

# from models.homographies import sample_homography
# from settings import DATA_PATH
# EXPER_PATH = 'logs'
DATA_PATH = 'datasets'


from imageio import imread
def load_as_float(path):
    return imread(path).astype(np.float32)/255

class PatchesDataset(object):
    default_config = {
        'dataset': 'hpatches',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': False
        }
    }

    def __init__(self, transform=None, **config):
        from utils.photometric import ImgAugTransform
        self.ImgAugTransform = ImgAugTransform

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.files = self._init_dataset(**self.config)
        sequence_set = []
        for (img, img_warped, mat_hom) in zip(self.files['image_paths'], self.files['warped_image_paths'], self.files['homography']):
            sample = {'image': img, 'warped_image': img_warped, 'homography': mat_hom}
            sequence_set.append(sample)
        self.samples = sequence_set
        self.transform = transform
        self.enable_photo = self.config['augmentation']['photometric']['enable']

        if config['preprocessing']['resize']:
            self.sizer = np.array(config['preprocessing']['resize'])
        pass

    def __getitem__(self, index):
        """

        :param index:
        :return:
            image:
                tensor (1,H,W)
            warped_image:
                tensor (1,H,W)
        """
        def _read_image(path):
            input_image = cv2.imread(path)
            image = input_image
            s = max(self.sizer /image.shape[:2])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[:int(self.sizer[0]/s),:int(self.sizer[1]/s)]
            image = cv2.resize(image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            return image
            # return input_image

        def _preprocess(image):
            # if image.ndim == 2:
            #     image = image[:,:, np.newaxis]
            if self.transform is not None:
                image = self.transform(image)
            return image

        # def _warp_image(image):
        #     H = sample_homography(tf.shape(image)[:2])
        #     warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
        #     return {'warped_im': warped_im, 'H': H}

        def _adapt_homography_to_preprocessing(image, H):
            # image = zip_data['image']
            # H = tf.cast(zip_data['homography'], tf.float32)
            # target_size = np.array(self.config['preprocessing']['resize'])
            s = max(self.sizer /image.shape[:2])
            # mat = np.array([[1,1,1/s], [1,1,1/s], [s,s,1]])
            mat = np.array([[1,1,s], [1,1,s], [1/s,1/s,1]])
            # down_scale = np.diag(np.array([1/s, 1/s, 1]))
            # up_scale = tf.diag(tf.stack([s, s, tf.constant(1.)]))
            # H = tf.matmul(up_scale, tf.matmul(H, down_scale))
            H = H*mat
            return H

        def imgPhotometric(img):
            """
            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.config['augmentation'])
            img = img[:,:,np.newaxis]
            img = augmentation(img)
            # cusAug = self.customizedTransform()
            # img = cusAug(img, **self.config['augmentation'])
            return img

        sample = self.samples[index]
        image_original = _read_image(sample['image'])
        image = image_original
        warped_image = _read_image(sample['warped_image'])
        ## add augmentation
        if self.enable_photo:
            # print(f"image: {image.shape}")
            image = imgPhotometric(image)
            # print(f"warped_image: {warped_image.shape}")
            warped_image = imgPhotometric(warped_image)

        image = _preprocess(image)
        warped_image = _preprocess(warped_image)

        to_numpy = False
        if to_numpy:
            image, warped_image = np.array(image), np.array(warped_image)
        homography = _adapt_homography_to_preprocessing(image_original, sample['homography'])
        sample = {'image': image, 'warped_image': warped_image,
                                    'homography': homography,
                                    'image_path': sample['image'],
                                    'warped_image_path': sample['warped_image']}
        return sample

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self, **config):
        dataset_folder = 'COCO/patches' if config['dataset'] == 'coco' else 'HPatches'
        base_path = Path(DATA_PATH, dataset_folder)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            num_images = 1 if config['dataset'] == 'coco' else 5
            file_ext = '.ppm' if config['dataset'] == 'hpatches' else '.jpg'
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
            warped_image_paths = warped_image_paths[:config['truncate']]
            homographies = homographies[:config['truncate']]
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies}
        return files



    # def _get_data(self, files, split_name, **config):
    #     def _read_image(path):
    #         return cv2.imread(path.decode('utf-8'))

    #     def _preprocess(image):
    #         tf.Tensor.set_shape(image, [None, None, 3])
    #         image = tf.image.rgb_to_grayscale(image)
    #         if config['preprocessing']['resize']:
    #             image = pipeline.ratio_preserving_resize(image,
    #                                                      **config['preprocessing'])
    #         return tf.to_float(image)

    #     def _warp_image(image):
    #         H = sample_homography(tf.shape(image)[:2])
    #         warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
    #         return {'warped_im': warped_im, 'H': H}

    #     def _adapt_homography_to_preprocessing(zip_data):
    #         image = zip_data['image']
    #         H = tf.cast(zip_data['homography'], tf.float32)
    #         target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
    #         s = tf.reduce_max(tf.cast(tf.divide(target_size,
    #                                             tf.shape(image)[:2]), tf.float32))
    #         down_scale = tf.diag(tf.stack([1/s, 1/s, tf.constant(1.)]))
    #         up_scale = tf.diag(tf.stack([s, s, tf.constant(1.)]))
    #         H = tf.matmul(up_scale, tf.matmul(H, down_scale))
    #         return H

    #     images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
    #     images = images.map(lambda path: tf.py_func(_read_image, [path], tf.uint8))
    #     homographies = tf.data.Dataset.from_tensor_slices(np.array(files['homography']))
    #     if config['preprocessing']['resize']:
    #         homographies = tf.data.Dataset.zip({'image': images,
    #                                             'homography': homographies})
    #         homographies = homographies.map(_adapt_homography_to_preprocessing)
    #     images = images.map(_preprocess)
    #     warped_images = tf.data.Dataset.from_tensor_slices(files['warped_image_paths'])
    #     warped_images = warped_images.map(lambda path: tf.py_func(_read_image,
    #                                                               [path],
    #                                                               tf.uint8))
    #     warped_images = warped_images.map(_preprocess)

    #     data = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
    #                                 'homography': homographies})

    #     return data
