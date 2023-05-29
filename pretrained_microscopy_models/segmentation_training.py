# Imports
import os
import shutil
import time
import numpy as np
import warnings
import numbers
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp

from itertools import cycle, product
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

from . import util
from . import io
from . import losses


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
    assert classes != 2, "If you are doing binary classification then set \
         classes=1 and the background class is implicit."
  
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
    if encoder_weights in ['micronet', 'image-micronet']:
        map = None if torch.cuda.is_available() else torch.device('cpu')
        url = util.get_pretrained_microscopynet_url(encoder, encoder_weights)
        model.encoder.load_state_dict(model_zoo.load_url(url, map_location=map))

    return model

def visualize_prediction_accuracy(prediction, truth, labels):
    out = np.zeros(truth.shape, dtype='uint8')
    trues = [np.all(truth == v, axis=-1) for v in labels]
    preds = [prediction[:,:,i] for i in range(prediction.shape[2])]
    for t, p in zip(trues, preds):
        out[t & p, :] = [255, 255, 255] # true positive
        out[t & ~p, :] = [255, 0, 255] # false negative
        out[~t & p, :] = [0, 255, 255] # false positive       
    return out

def load_segmentation_model(model_path, classes):
    """Load a segmentation model from saved state path

    Args:
        model_path (str): Path to saved state
        classes (int): number of segmentation classes

    Returns:
        nn.module: PyTorch segmentation model
    """
    state = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    architecture = state['architecture']
    encoder = state['encoder']
    model=create_segmentation_model(architecture, encoder, None, classes)
    model.load_state_dict(util.remove_module_from_state_dict(state['state_dict']))
    model.to(device)
    model.eval()
    try:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, 'imagenet')
    except ValueError:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, 'imagenet+5k')
    return model, preprocessing_fn

def extract_patches(arr, patch_shape=8, extraction_step=1):
    #THIS FUNCTION COMES FROM AN OLD VERSION OF SCIKIT-LEARN
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches

# https://github.com/choosehappy/PytorchDigitalPathology
def segmentation_models_inference(io, model, preprocessing_fn, device = None, batch_size = 8, patch_size = 512,
                                  num_classes=3, probabilities=None):

    # This will not output the first class and assumes that the first class is wherever the other classes are not!
    try:
        io = preprocessing_fn(io)
    except AttributeError:
        io = preprocessing_fn(np.array(io))
        
    io_shape_orig = np.array(io.shape)
    stride_size = patch_size // 2
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # add half the stride as padding around the image, so that we can crop it away later
    io = np.pad(io, [(stride_size // 2, stride_size // 2), (stride_size // 2, stride_size // 2), (0, 0)],
                mode="reflect")

    io_shape_wpad = np.array(io.shape)

    # pad to match an exact multiple of unet patch size, otherwise last row/column are lost
    npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
    npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

    io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arr_out = extract_patches(io, (patch_size, patch_size, 3), stride_size)

    arr_out_shape = arr_out.shape
    arr_out = arr_out.reshape(-1, patch_size, patch_size, 3)

    # in case we have a large network, lets cut the list of tiles into batches
    output = np.zeros((0, num_classes, patch_size, patch_size))
    def divide_batch(l, n): 
        for i in range(0, l.shape[0], n):  
            yield l[i:i + n,::] 
    for batch_arr in divide_batch(arr_out, batch_size):
        arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2).astype('float32')).to(device)

        # ---- get results
        output_batch = model.predict(arr_out_gpu)

        # --- pull from GPU and append to rest of output
        if probabilities is None:
            output_batch = output_batch.detach().cpu().numpy().round()
        else:
            output_batch = output_batch.detach().cpu().numpy()

        output = np.append(output, output_batch, axis=0)

    output = output.transpose((0, 2, 3, 1))

    # turn from a single list into a matrix of tiles
    output = output.reshape(arr_out_shape[0], arr_out_shape[1], patch_size, patch_size, output.shape[3])

    # remove the padding from each tile, we only keep the center
    output = output[:, :, stride_size // 2:-stride_size // 2, stride_size // 2:-stride_size // 2, :]

    # turn all the tiles into an image
    output = np.concatenate(np.concatenate(output, 1), 1)

    # incase there was extra padding to get a multiple of patch size, remove that as well
    output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :]  # remove paddind, crop back
    if probabilities is None:
        if num_classes == 1:
            return output.astype('bool')
        else:
            return output[:, :, 1:].astype('bool')
    else:
        for i in range(num_classes-1): #don't care about background class
            output[:,:,i+1] = output[:,:,i+1] > probabilities[i]
        return output[:, :, 1:].astype('bool')


def train_segmentation_model(model,
                             architecture,
                             encoder,
                             train_dataset,
                             validation_dataset,
                             class_values, 
                             loss=None,
                             epochs=None, 
                             patience=30,
                             device='cuda', 
                             lr=2e-4, 
                             lr_decay=0.00, 
                             batch_size=20, 
                             val_batch_size=12, 
                             num_workers=0, 
                             save_folder='./', 
                             save_name=None,
                             multi_gpu=False):

    # setup and check parameters
    assert len(class_values) != 2, "Two classes is binary classification.  Just specify the posative class value."
    assert patience is not None or epochs is not None, "Need to set patience or epochs to define a stopping point." 
    epochs = 1e10 if epochs is None else epochs # this will basically never be reached.
    patience = 1e10 if patience is None else patience # this will basically never be reached.

    # load or create model
        # TODO need to reload the optimizer.
    if type(model) is dict: # passed the state back for restarting
        state = model #load state dictionary
        architecture = state['architecture']
        encoder = state['encoder']
        # create empty model
        model=create_segmentation_model(architecture, encoder, None, len(class_values))
        # load saved weights
        model.load_state_dict(util.remove_module_from_state_dict(state['state_dict']))
    elif type(model) is str: # passed saved model state for restarting
        state = torch.load(model) 
        architecture = state['architecture']
        encoder = state['encoder']
        model=create_segmentation_model(architecture, encoder, None, len(class_values))
        model.load_state_dict(util.remove_module_from_state_dict(state['state_dict']))
    else: # Fresh PyTorch model for segmentation
        state = {
            'architecture': architecture,
            'encoder': encoder,
            'train_loss': [],
            'valid_loss': [],
            'train_iou': [],
            'valid_iou': [],
            'max_score': 0,
            'class_values': class_values
        }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
       
    
    if multi_gpu:
        model = torch.nn.DataParallel(model).cuda()

        
    # create training dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)  
    valid_loader = DataLoader(validation_dataset, batch_size=val_batch_size, 
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    
    
    loss = losses.DiceBCELoss(weight=0.7) if loss is None else loss

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
            if save_name is not None:
                shutil.copyfile(os.path.join(save_folder, 'model_best.pth.tar'), 
                                os.path.join(save_folder, save_name))
            return state

            
        # Use early stopping if there has not been improvment in a while
        if patience_step >= patience:
            print('\n\nTraining done!  No improvement in {} epochs. Saving final model'.format(patience))
            if save_name is not None:
                shutil.copyfile(os.path.join(save_folder, 'model_best.pth.tar'), 
                                os.path.join(save_folder, save_name))
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
