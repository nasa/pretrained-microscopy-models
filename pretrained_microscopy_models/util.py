import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.utils.model_zoo as model_zoo


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

def get_segmentation_model(architecture, 
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
        url = get_pretrained_microscopynet_url(encoder, encoder_weights)
        model.encoder.load_state_dict(model_zoo.load_url(url, map_location=map))

    return model

