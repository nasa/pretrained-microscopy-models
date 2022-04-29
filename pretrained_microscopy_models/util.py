import functools
from webbrowser import get
import matplotlib.pyplot as plt
from collections import OrderedDict
import segmentation_models_pytorch as smp

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# dot dict for args
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_pretrained_microscopynet_url(encoder, encoder_weights, version=1.1, 
                                     self_supervision=''):
    """Get the url to download the specified pretrained encoder.

    Args:
        encoder (str): pretrained encoder model name (e.g. resnet50)
        encoder_weights (str): pretraining dataset, either 'microscopynet' or 
            'imagenet-microscopynet' with the latter indicating the encoder
            was first pretrained on imagenet and then finetuned on microscopynet
        version (float): model version to use, defaults to latest. 
            Current options are 1.0 or 1.1.
        self_supervision (str): self-supervision method used. If self-supervision
            was not used set to '' (which is default).

    Returns:
        str: url to download the pretrained model
    """

    # only resnet50/micronet has version 1.1 so I'm not going to overcomplicate this right now.
    if encoder != 'resnet50' or encoder_weights != 'micronet':
        version = 1.0

    # setup self-supervision
    if self_supervision != '':
        version = 1.0
        self_supervision = '_' + self_supervision

    # correct for name change for URL
    if encoder_weights == 'micronet':
        encoder_weights = 'microscopynet'
    elif encoder_weights == 'image-micronet':
        encoder_weights = 'imagenet-microscopynet'
    else:
        raise ValueError("encoder_weights must be 'micronet' or 'image-micronet'")

    # get url
    url_base = 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/'
    url_end = '_v%s.pth.tar' %str(version)
    return url_base + f'{encoder}{self_supervision}_pretrained_{encoder_weights}' + url_end


def remove_module_from_state_dict(state_dict):
    """Removes 'module.' from nn.Parallel models.  
    If module does not exist it just returns the state dict"""
    if list(state_dict.keys())[0].startswith('module'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v           
        return new_state_dict
    elif list(state_dict.keys())[0].startswith('features.module'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[:9] + k[9+7:] # remove `module.`
            if name.startswith('features.'):
                new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def get_special_preprocessing_fn(mean= [0.4723, 0.4599, 0.4468], std = [0.1684, 0.1575, 0.1675]):
    params = {"mean": mean,
              "std": std}
    return functools.partial(smp.encoders._preprocessing.preprocess_input, **params)

# debugging
if __name__ == '__main__':
    print(get_pretrained_microscopynet_url('se_resnet50', 'micronet', version=1.1))
    print(get_pretrained_microscopynet_url('resnet50', 'micronet', version=1.0))
    print(get_pretrained_microscopynet_url('resnet50', 'micronet'))