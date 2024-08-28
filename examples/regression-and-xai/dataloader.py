''' dataloader class that contains helper functions for preprocessing and augmenting images for model training '''

import os
import numpy as np
import pandas as pd
import random

import albumentations as albu
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# shift channel to front of tensor
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# preprocess image in accordance with encoder
def get_preprocessing(encoder):
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor)
    ]
    return albu.Compose(_transform)

class Dataset(BaseDataset):

    def __init__(
            self, 
            images,
            dataframe,
            features,
            augmentation=None, 
            preprocessing=None,
    ):

        self.dataframe = dataframe
        self.features = features
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # create list of image paths
        if type(images) is list:
            self.images_fps = images
        
    def __getitem__(self, i):
        
        # read image into array
        image = np.load(self.images_fps[i])
        
        image_id = self.images_fps[i]
        image_id = os.path.basename(image_id).split('_')[0] + '.TIF'
        
        # query accompanying metadata for image
        target_row = self.dataframe[self.dataframe['file_path'] == image_id]
        target_row = target_row[self.features]
        targets = target_row.to_numpy()[0, 1:].astype('float32')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image, targets

    def __len__(self):
        return len(self.images_fps)
    
# create dataset using custom Dataset class
def create_dataset(images, df, features, encoder, set_size='full'):
    train_df = pd.DataFrame({'path': images})
    train_df['parent_id'] = train_df['path'].apply(os.path.basename).str.split('-').str[1:3].str.join('-')
    
    if set_size == '5':
        parent_ids = ['RCI-007', 'RCI-N025', 'RCI-008', 'RCI-N024', 'RCI-003']
    elif set_size == '3':
        parent_ids = ['RCI-007', 'RCI-008', 'RCI-003']
    elif set_size == '1':
        parent_ids = ['RCI-008']
    else:
        parent_ids = train_df['parent_id'].tolist()
                               
    train_df = train_df[train_df['parent_id'].isin(parent_ids)]
    images = train_df['path'].tolist()
    
    dataset = Dataset(
        images=images,
        dataframe=df,
        features=features,
        augmentation=None,
        preprocessing=get_preprocessing(encoder)
    )

    return dataset

# dataloader to batch datset for training
def create_dataloader(images, df, features, encoder, set_size='full', batch_size=16):
    dataset = create_dataset(images, df, features, encoder, set_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    return dataloader

class PatchDataset(BaseDataset):

    def __init__(
            self, 
            images,
            centers,
            preprocessing=None,
    ):
        
        self.preprocessing = preprocessing

        # create list of image paths
        if type(images) is list:
            self.images_fps = images
            self.centers_fps = centers
            
    def __getitem__(self, i):
        
        # read in image
        image = self.images_fps[i]
        if image.ndim == 2:
            image = gray2rgb(image)
        
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
            
        center = self.centers_fps[i]
            
        return image, center

    def __len__(self):
        return len(self.images_fps)
    
# create dataset using custom Dataset class
def create_patch_dataset(images, centers, encoder):
    dataset = PatchDataset(
        images=images,
        centers=centers,
        preprocessing=get_preprocessing(encoder)
    )

    return dataset