import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
# from settings import EXPER_PATH
EXPER_PATH = 'logs'

from imgaug import augmenters as iaa
import imgaug as ia

# class ImgAugTransform:
#     def __init__(self, **config):
#         if config['photometric']['enable']:
#             self.aug = iaa.Sequential([
#                 # iaa.Scale((224, 224)),
#                 iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
#                 # iaa.Fliplr(0.5),
#                 # iaa.Affine(rotate=(-20, 20), mode='symmetric'),
#                 iaa.Sometimes(0.25,
#                               iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                          iaa.CoarseDropout(0.1, size_percent=0.5)])),
#                 iaa.Sometimes(0.25,
#                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
#                               )
#                 # iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=False)
#             ])
#         else:
#             self.aug = iaa.Sequential([
#                 iaa.Noop(),
#             ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)

# from utils.loader import get_save_path
def get_save_path(output_dir):
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
   base_seed = torch.IntTensor(1).random_().item()
   # print(worker_id, base_seed)
   np.random.seed(base_seed + worker_id)


def dataLoader(config, dataset='syn', warp_input=False, train=True, val=True):
    import torchvision.transforms as transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    if dataset == 'syn':
        from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset
    else:
        Dataset = get_module('datasets', dataset)
        # from datasets.coco import Coco as Dataset

    train_set = Dataset(
        transform=data_transforms['train'],
        task = 'train',
        **config['data'],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['model']['batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=8,
        worker_init_fn=worker_init_fn
    )
    val_set = Dataset(
        transform=data_transforms['train'],
        task = 'val',
        **config['data'],
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config['model']['eval_batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=8,
        worker_init_fn=worker_init_fn
    )
    # val_set, val_loader = None, None
    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}

def dataLoader_test(config, dataset='syn', warp_input=False, export_task='train',
            shuffle=False, device='cpu'):
    import torchvision.transforms as transforms
    if device == 'cpu':
        num_workers = 0
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }
    test_loader = None
    if dataset == 'syn':
        from datasets.SyntheticDataset import SyntheticDataset
        test_set = SyntheticDataset(
            transform=data_transforms['test'],
            train=False,
            warp_input=warp_input,
            getPts=True,
            seed=1,
            **config['data'],
        )
    elif dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset
        if config['data']['preprocessing']['resize']:
            size = config['data']['preprocessing']['resize']
        test_set = PatchesDataset(
            transform=data_transforms['test'],
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )
    elif dataset == 'coco':
        from datasets.coco import Coco
        test_set = Coco(
            export=True,
            task=export_task,
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )
    # elif dataset == 'Kitti' or 'Tum':
    else:
        # from datasets.Kitti import Kitti
        logging.info(f"load dataset from : {dataset}")
        Dataset = get_module('datasets', dataset)
        test_set = Dataset(
            export=True,
            task=export_task,
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn

        )
    return {'test_set': test_set, 'test_loader': test_loader}

def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)

def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)

def modelLoader(model='SuperPointNet', **options):
    # create model
    logging.info("=> creating model: %s", model)
    net = get_model(model)
    net = net(**options)
    return net


# mode: 'full' means the formats include the optimizer and epoch
# full_path: if not full path, we need to go through another helper function
def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    # load checkpoint
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    if mode == 'full':
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch = checkpoint['epoch']
        epoch = checkpoint['n_iter']
#         epoch = 0
    else:
        net.load_state_dict(checkpoint)
        # net.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage))
    return net, optimizer, epoch

if __name__ == '__main__':
    net = modelLoader(model='SuperPointNet')

