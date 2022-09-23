"""
Contains various utility functions for PyTorch model training and saving
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to save
    target_dir : str
        target directory to save the model in
    model_name : str
        name of the model. Should include either ".pth" or ".pt" as file extension
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

# TODO add plot loss curves function