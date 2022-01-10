import cv2
import os
import numpy as np

from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    Modified from https://github.com/qubvel/segmentation_models.pytorch
    
    
    Args:
        images_dir (str or list): path to images folder or list of images
        masks_dir (str): path to segmentation masks folder or list of images
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
            images, 
            masks, 
            class_values,
            augmentation=None, 
            preprocessing=None,
    ):

        self.class_values = class_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # create list of image paths
        if type(images) is list:
            self.images_fps = images
            self.masks_fps = masks
        else:
            self.ids = os.listdir(images)
            self.images_fps = [os.path.join(images, image_id) for image_id in self.ids]
            # create list of annotation paths (MAY NEED TO ADJUST THIS LOGIC)
            self.masks_fps = [os.path.join(masks, image_id.replace('.tif', '_mask.tif')) for image_id in self.ids]
        
        
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

    def __len__(self):
        return len(self.images_fps)