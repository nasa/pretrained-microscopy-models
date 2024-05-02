import time
import copy
import sys

import torch
import torch.nn as nn
import pretrainedmodels
import segmentation_models_pytorch as smp
import pretrained_microscopy_models as pmm
import torch.utils.model_zoo as model_zoo

def generate_model(encoder, encoder_weights):
    
    weights_map = {'imagenet': 'imagenet', 'micronet': None, 'image-micronet': None, 'scratch': None}
    
    # load model from appropiate host service
    if encoder == 'resnext101_32x8d':
        model = torch.hub.load('pytorch/vision:v0.10.0', encoder, pretrained=weights_map[encoder_weights])
    else:
        model = pretrainedmodels.__dict__[encoder](num_classes=1000, pretrained=weights_map[encoder_weights])
    
    # load pretrained micronet or image-micronet weights
    if encoder_weights == 'micronet' or encoder_weights == 'image-micronet':
        url = pmm.util.get_pretrained_microscopynet_url(encoder, encoder_weights)
        weights = model_zoo.load_url(url)
    
        # custom feature naming for mismatched weight names
        if encoder == 'vgg16_bn':
            model._modules["features"] = model._modules.pop("_features")
            model.load_state_dict(weights, strict=False)
            model._modules["_features"] = model._modules.pop("features")
        else:
            model.load_state_dict(weights, strict=False)
    
    # convert last layers to dense layers for regression
    if encoder == 'resnext101_32x8d':
        num_ftrs = model.fc.in_features
        
        model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 100),
                    nn.ReLU(),
                    nn.Linear(100, 1)
                )
        
        return model
    else:
        num_ftrs = model.last_linear.in_features

        model.last_linear = nn.Sequential(
                        nn.Linear(num_ftrs, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 100),
                        nn.ReLU(),
                        nn.Linear(100, 1)
                    )
        
        return model

def freeze_base_weights(model):
    for name, param in model.named_parameters():
        if name.startswith('last_linear') or name.startswith('fc'):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model

def train_model(model, device, dataloaders, criterion, optimizer, scheduler, patience, patience_start, num_epochs=20):

    # track time per epoch
    since = time.time()
    
    # metrics to record
    train_loss_history = []
    val_loss_history = []
    
    # copy best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.maxsize
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print('Training phase:')
                model.train()
            else:
                print('Validation phase:')
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                
                if epoch_loss < best_loss:
                    early_stopping_counter = 0
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    if epoch > patience_start:
                        early_stopping_counter += 1
                        print(f'No improvement - incrementing early stopping count to: {early_stopping_counter}')
        
        if scheduler is not None:
            scheduler.step()
        
        if early_stopping_counter == patience:
            break
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    histories = [train_loss_history, val_loss_history]

    return model, histories