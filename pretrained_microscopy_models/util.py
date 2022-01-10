import matplotlib.pyplot as plt
from collections import OrderedDict

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

def get_pretrained_microscopynet_url(encoder, encoder_weights):
    """Get the url to download the specified pretrained encoder.

    Args:
        encoder (str): pretrained encoder model name (e.g. resnet50)
        encoder_weights (str): pretraining dataset, either 'microscopynet' or 
            'imagenet-microscopynet' with the latter indicating the encoder
            was first pretrained on imagenet and then finetuned on microscopynet

    Returns:
        str: url to download the pretrained model
    """
    # correct for name change for URL
    if encoder_weights == 'micronet':
        encoder_weights = 'microscopynet'
    elif encoder_weights == 'image-micronet':
        encoder_weights = 'imagenet-microscopynet'
    else:
        raise ValueError("encoder_weights must be 'micronet' or 'image-micronet'")

    # get url
    url_base = 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/'
    url_end = '_v1.0.pth.tar'
    return url_base + f'{encoder}_pretrained_{encoder_weights}' + url_end


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
