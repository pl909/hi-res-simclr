import torch.nn as nn
from torchvision import models

from model_head import *


# This code creates a custom High-dimension architecture of DenseNet121 by linking a model_head to a default Densenet w/ top layer removed. See model_head.py

# Can choose input size of the model (448 or 896)

# Size = input size. weights = pretrained or not. classes = # of classes

def HiDenseNet(size, weights, classes):
    def truncated_densenet121(weights="Dense121"):
        # Load the DenseNet121 model
        if weights == "Dense121":
            base_model = models.densenet121(pretrained=True)
            print('pretrained')
        elif weights == "None":
            base_model = models.densenet121(pretrained=False)
        else:
            raise ValueError('weights should be either: "Dense121" or "None"')

        # Remove the classifier (the "top" of the model)
        base_model.classifier = nn.Identity()

        # Remove the initial layers to match the truncation of ResNet50.
        truncated_features = nn.Sequential(*list(base_model.features.children())[4:])
        base_model.features = truncated_features

        return base_model

    # Selects architecture based on input size (448 by 448) or (896 by 896)
    if size == 448:
        hi_res_head = DenseNet448()
    elif size == 896:
        hi_res_head = DenseNet896()
    else:
        raise ValueError('size should be an integer value of: 448 or 896')

    if not isinstance(classes, int):
        raise ValueError('classes must be an integer')

    #
    base_model = truncated_densenet121(weights)

    # Combining HiResNet head with DenseNet121 base
    model = nn.Sequential(hi_res_head, base_model)

    # Adding final layers (currently commented out because class downstreaming is unnecessary).
    '''
    model.add_module('MaxPool2D', nn.MaxPool2d((4, 4)))
    model.add_module('Flatten', nn.Flatten())
    model.add_module('Dense1024', nn.Linear(1024, 1024))
    model.add_module('ReLU', nn.ReLU())
    model.add_module('Dropout1', nn.Dropout(0.2))
    model.add_module('Dense512', nn.Linear(1024, 512))
    model.add_module('ReLU2', nn.ReLU())
    model.add_module('Dropout2', nn.Dropout(0.2))
    model.add_module('FinalDense', nn.Linear(512, classes))
    model.add_module('Softmax', nn.Softmax(dim=1))
    '''
    return model