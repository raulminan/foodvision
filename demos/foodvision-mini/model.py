import torch
import torchvision

from torch import nn
from typing import Tuple

def create_effnetb2_model(
    num_classes: int = 3, 
    seed: int = 42) -> Tuple[torch.nn.Module, torchvision.transforms.Compose]:
    """Creates an EfficientNetB2 feature extractor model and transforms

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the classifier head, by default 3
    seed : int, optional
        random seed value, by default 42

    Returns
    -------
    Tuple[torch.nn.Module, torchvision.transforms.Compose]
        Tuple[EffnetB2 feature extractor model, EffNetb2 image transforms]
    """
    # Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # change classifier head
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model, transforms
