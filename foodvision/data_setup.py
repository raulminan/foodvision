"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""
import os

from typing import Tuple, List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloader(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, List[str]]:
    
    """Creates training and testing DataLoaders
    
    Takes in a training directory and testing directory path and turns
    then into PyTorch Datasets and then into PyTorch DataLoaders

    Parameters
    ----------
    train_dir : str
        Path to training directory
    test_dir : str
        Path to testing directory
    transform : transforms.Compose
        torchvision transforms to perform on training and testing data
    batch_size : int
        number of samples per batch in each DataLoader
    num_workers : int, optional
        number of workers (parallel processes) per DataLoader, by default NUM_WORKERS
    
    Returns
    -------
    Tuple[DataLoader, DataLoader, List[str]]
        A tuple of (train_dataloader, test_dataloader, class_names)
        Where class_names is a list of the target classes
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )
    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )
    
    class_names = train_data.classes
    
    # create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader, class_names
