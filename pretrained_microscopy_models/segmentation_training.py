# Imports
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp

from itertools import cycle, product
from collections import OrderedDict
from torch.utils.data import DataLoader


import util
import io
import losses


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

def create_segmentation_model(architecture, 
    encoder, 
    encoder_weights, 
    classes, 
    activation=None
):
    """Returns a segmentation model with the specified architecture and encoder 
    backbone. The encoder can be pre-trained with ImageNet, MicroNet,  
    ImageNet --> MicroNet (ImageMicroNet), or no pretraining.

    Args:
        architecture (str): Segmentation architecture available in 
            segmentation_models_pytorch. E.g. 'Unet', 'UnetPlusPlus', 'Linknet',
            'FPN', 'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus'
        encoder (str): One of the available encoder backbones in 
            segmentation_models_pytorch such as 'ResNet50' or 'efficientnet-b3'
        encoder_weights (str): The dataset that the encoder was pre-trained on.
            One of ['micronet', 'image-micronet', 'imagenet', 'None']
        classes (int): number of output classes to segment
        activation (str, optional): Activation function of the last layer. 
            If None is set based on number of classes. Defaults to None.

    Returns:
        nn.Module: PyTorch model for segmentation
    """

    # setup and check parameters
    assert classes != 2, "Two classes is binary classification.  \
        Just specify the posative class value"
  
    if activation is None:
        activation = 'softmax2d' if classes > 1 else 'sigmoid' 

    initial_weights = 'imagenet' if encoder_weights == 'imagenet' else None

    if initial_weights == 'imagenet' and \
        encoder in ['dpn68b',  'dpn92', 'dpn137', 'dpn107']:
        initial_weights = 'imagenet+5k'

    # create the model
    try:
        model = getattr(smp, architecture)(
            encoder_name=encoder, 
            encoder_weights=initial_weights,
            classes=classes,
            activation=activation)
    except ValueError:
        raise ValueError('%s does not support dilated mode needed for %s.' %(encoder, architecture))

    # load pretrained weights 
    if encoder_weights in ['micronet', 'imagemicronet']:
        map = None if torch.cuda.is_available() else torch.device('cpu')
        url = util.get_pretrained_microscopynet_url(encoder, encoder_weights)
        model.encoder.load_state_dict(model_zoo.load_url(url, map_location=map))

    return model

def train_segmentation_model(model,
                             train_dataset,
                             validation_dataset,
                             class_values, 
                             epochs=None, #ADD LOGIC TO STOP AFTER THIS MANY EPOCHS
                             patience=30,
                             device='cuda', 
                             lr=2e-4, 
                             lr_decay=0.00, 
                             batch_size=20, 
                             val_batch_size=12, 
                             num_workers=0, 
                             save_folder='./', 
                             multi_gpu=False):

    # setup and check parameters
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    assert len(class_values) != 2, "Two classes is binary classification.  Just specify the posative class value"
            
    if multi_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # create dataloaders
    try:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    except ValueError:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet+5k')
        
   
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)  
    valid_loader = DataLoader(validation_dataset, batch_size=val_batch_size, 
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    
    loss = losses.DiceBCELoss(weight=0.7)

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

    state = {
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
        torch.save(state, os.path.join(save_folder, 'checkpoint.pth.tar'))
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            patience_step = 0
            max_score = valid_logs['iou_score']
            best_epoch = epoch + 1
            torch.save(state, os.path.join(save_folder, 'model_best.pth.tar'))
            print('Best model saved!')
        else:
            patience_step += 1

        
        # Increment the epoch and decay the learning rate
        epoch += 1
        lr = optimizer.param_groups[0]['lr'] * (1-lr_decay)
        optimizer.param_groups[0]['lr'] = lr
        
        if epoch >= epochs:
            print('\n\nTraining done! Saving final model')
            return state

            
        # Use early stopping if there has not been improvment in a while
        if patience_step >= patience:
            print('\n\nTraining done!  No improvement in {} epochs. Saving final model'.format(patience))
            return state

    

# if __name__ == '__main__':
#     encoder = 'resnet34'
#     encoder_weights = 'microscopynet'
#     decoder = 'Unet'
#     class_values = {'matrix': [0,0,0],
#                'secondary': [255,0,0],
#                'tertiary' : [0,0,255]}

#     activation = 'softmax2d' if len(class_values) > 1 else 'sigmoid' #'softmax2d' for multicalss segmentation

#     initial_weights = 'imagenet' if 'imagenet' in encoder_weights else None
#     # load model
       
#     if initial_weights == 'imagenet' and ('dpn68b' in encoder or 'dpn92' in encoder or 'dpn137' in encoder
#                                          or 'dpn107' in encoder):
#         initial_weights = 'imagenet+5k'
    
#     try:
#         model = getattr(smp, decoder)(encoder_name=encoder, 
#                                       encoder_weights=initial_weights,
#                                       classes=len(class_values),
#                                       activation=activation)
#     except ValueError: #certain encoders do not support encoder dilation
#         if decoder == 'DeepLabV3Plus':
#             print('\n\n%s does not support dilated mode needed for %s. Skipping.\n\n' %(encoder, decoder))
#             exit()
#         else:
#             model = getattr(smp, decoder)(encoder_name=encoder, 
#                                           encoder_weights=initial_weights,
#                                           classes=len(class_values),
#                                           activation=activation,
#                                           encoder_dilation=False)



#     url = util.get_pretrained_microscopynet_url(encoder, encoder_weights)
#     model.encoder.load_state_dict(model_zoo.load_url(url))
