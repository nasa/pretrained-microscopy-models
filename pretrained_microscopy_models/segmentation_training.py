# Imports
import os
import csv
import shutil
import glob
import time
import random
random.seed(0)

from itertools import cycle, product
from collections import OrderedDict

import numpy as np
np.random.seed(0)
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as albu
import seaborn as sns

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.utils.model_zoo as model_zoo

import segmentation_models_pytorch as smp

import util

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    Modified from https://github.com/qubvel/segmentation_models.pytorch
    
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (dict): values of classes to extract from segmentation mask. 
            Each dictionary value can be an integer or list that specifies the mask
            values that belong to the class specified by the corresponding dictionary key.
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_values,
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('.tif', '_mask.tif')) for image_id in self.ids]
        
        self.class_values = class_values
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], 1)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # extract certain classes from mask (e.g. cars)
        masks = [np.all(mask == v, axis=-1) for v in self.class_values.values()]
        masks[0] = ~masks[1] & ~masks[2]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask

def get_training_augmentation():
    train_transform = [

        albu.Flip(p=0.75),
        albu.RandomRotate90(p=1),
        
        albu.IAAAdditiveGaussianNoise(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1, limit=0.25),
                albu.RandomGamma(p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                #albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1, limit=0.3),
                albu.HueSaturationValue(p=1),
            ],
            p=0.50,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #albu.Resize(height,width)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
        
    def __len__(self):
        return len(self.ids)

def train_model(decoder, encoder, encoder_weights, class_values, device='cuda', lr=2e-4, lr_decay=0.00,
               batch_size=20, val_batch_size=12, num_workers=0, patience=30, save_folder='./', multi_gpu=False, step_lr=None):
    print(decoder, encoder, encoder_weights)
    # setup and check parameters
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    assert len(class_values) != 2, "Two classes is binary classification.  Just specify the posative class value"
    activation = 'softmax2d' if len(class_values) > 1 else 'sigmoid' #'softmax2d' for multicalss segmentation
    initial_weights = None if encoder_weights is None else 'imagenet'
    # load model
    
    if encoder in ['efficientnet-b5', 'senet154']:
        batch_size = 12
    
    if initial_weights == 'imagenet' and ('dpn68b' in encoder or 'dpn92' in encoder or 'dpn137' in encoder
                                         or 'dpn107' in encoder):
        initial_weights = 'imagenet+5k'
    
    try:
        model = getattr(smp, decoder)(encoder_name=encoder, 
                                      encoder_weights=initial_weights,
                                      classes=len(class_values),
                                      activation=activation)
    except ValueError: #certain encoders do not support encoder dilation
        if decoder == 'DeepLabV3Plus':
            print('\n\n%s does not support dilated mode needed for %s. Skipping.\n\n' %(encoder, decoder))
            return
        else:
            model = getattr(smp, decoder)(encoder_name=encoder, 
                                          encoder_weights=initial_weights,
                                          classes=len(class_values),
                                          activation=activation,
                                          encoder_dilation=False)
        
        
#     # load pretrained weights 
#     if encoder_weights in ['microscopynet', 'microscopynet_fromscratch']:
#         # load the saved state dict
#         try:
#             if encoder_weights == 'microscopynet':
#                 path = os.path.join(MICRO_MODELS_DIR, microscopynet_weights[encoder])
#             else:
#                 path = os.path.join(MICRO_MODELS_DIR, microscopynet_fromscratch_weights[encoder])
#         except KeyError:
#             print('\n\nNo pretrained %s weights for %s encoder!!\n\n' %(encoder, encoder_weights))
#             return
#         state_dict = torch.load(path)['state_dict']    
        
#         # remove module. from keys if trained with DataParallel
#         state_dict = remove_module_from_state_dict(state_dict)
        
#         # fix last_linear.bias and last_linear weigths for xception pretrained
#         if encoder == 'xception':
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 name = k.replace('last_linear', 'fc')
#                 new_state_dict[name] = v    
#             state_dict = new_state_dict
        path = Path('pretrained_microscopynet_models' , 'inceptionresnetv2_pretrained_microscopynet.pth.tar')
        model.encoder.load_state_dict(torch.load(path))
        
    if multi_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # create dataloaders
    try:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    except ValueError:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet+5k')
        
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_values=class_values,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_values=class_values,
    )
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)  
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, 
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    
    loss = DiceBCELoss(weight=0.7)

    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),])
    
    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )
    
    patience_step = 0
    max_score = 0
    best_epoch = 0
    epoch = 0
    t0 = time.time()

    state = {'encoder': encoder,
             'decoder': decoder,
             'train_loss': [],
             'valid_loss': [],
             'train_iou': [],
             'valid_iou': [],
             'max_score': 0,
             'class_values': class_values
            }
    
    while True:
        t = time.time() - t0
        print('\nEpoch: {}, lr: {:0.8f}, time: {:0.2f} seconds, patience step: {}, best iou: {:0.4f}'.format(
            epoch, lr, t, patience_step, max_score))
        t0 = time.time()
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # update the state
        state['epoch'] = epoch + 1
        state['state_dict'] = model.state_dict()
        state['optimizer'] = optimizer.state_dict()
        state['train_loss'].append(train_logs['DiceBCELoss'])
        state['valid_loss'].append(valid_logs['DiceBCELoss'])
        state['train_iou'].append(train_logs['iou_score'])
        state['valid_iou'].append(valid_logs['iou_score'])
        
        # save the model
        #torch.save(state, os.path.join(save_folder, 'checkpoint.pth.tar'))
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            patience_step = 0
            max_score = valid_logs['iou_score']
            best_epoch = epoch + 1
#             shutil.copyfile(os.path.join(save_folder, 'checkpoint.pth.tar'), 
#                             os.path.join(save_folder, 'model_best.pth.tar'))
            torch.save(state, os.path.join(save_folder, 'model_best.pth.tar'))
            print('Best model saved!')
        
        else:
            patience_step += 1

        
        # Increment the epoch and decay the learning rate
        epoch += 1
        lr = optimizer.param_groups[0]['lr'] * (1-lr_decay)
        optimizer.param_groups[0]['lr'] = lr
        
        if epoch == 60:
            lr = 5e-4
            optimizer.param_groups[0]['lr'] = lr

            
        # Use early stopping if there has not been improvment in a while
        if patience_step > patience:
            print('\n\nTraining done!  No improvement in {} epochs. Saving final model'.format(patience))
            shutil.copyfile(os.path.join(save_folder, 'model_best.pth.tar'), 
                            os.path.join(save_folder, '{}__{}__{}__{}__{:.3f}.pth.tar'.format(
                                decoder, encoder, encoder_weights, best_epoch, max_score)))
            break

    

if __name__ == '__main__':
    encoder = 'resnet34'
    encoder_weights = 'microscopynet'
    decoder = 'Unet'
    class_values = {'matrix': [0,0,0],
               'secondary': [255,0,0],
               'tertiary' : [0,0,255]}

    activation = 'softmax2d' if len(class_values) > 1 else 'sigmoid' #'softmax2d' for multicalss segmentation

    initial_weights = 'imagenet' if 'imagenet' in encoder_weights else None
    # load model
       
    if initial_weights == 'imagenet' and ('dpn68b' in encoder or 'dpn92' in encoder or 'dpn137' in encoder
                                         or 'dpn107' in encoder):
        initial_weights = 'imagenet+5k'
    
    try:
        model = getattr(smp, decoder)(encoder_name=encoder, 
                                      encoder_weights=initial_weights,
                                      classes=len(class_values),
                                      activation=activation)
    except ValueError: #certain encoders do not support encoder dilation
        if decoder == 'DeepLabV3Plus':
            print('\n\n%s does not support dilated mode needed for %s. Skipping.\n\n' %(encoder, decoder))
            exit()
        else:
            model = getattr(smp, decoder)(encoder_name=encoder, 
                                          encoder_weights=initial_weights,
                                          classes=len(class_values),
                                          activation=activation,
                                          encoder_dilation=False)



    url = util.get_pretrained_microscopynet_url(encoder, encoder_weights)
    model.encoder.load_state_dict(model_zoo.load_url(url))
