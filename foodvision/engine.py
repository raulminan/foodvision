""" 
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    
    """Trains a PyTorch model for a single epoch

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model to be trained
    dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be trained on
    loss_fn : torch.nn.Module
        A PyTorch loss function to minimize
    optimizer : torch.optim.Optimizer
        A PyTorch optimizer to help minimize the loss function
    device : torch.device
        A target device to compute on (e.g "cuda" or "cpu")

    Returns
    -------
    Tuple[float, float]
        A tuple of training loss and training accuracy metrics
        (train_loss, train_accuracy)
    """
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        
        # 2. Calculate loss and acumulate
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
    # average metrics
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    
    """Tests a PyTorch model for a single epoch
    
    Turns a target PyTorch model to "eval" mode then performs a forward
    pass on a testing dataet

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model to be testes
    dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be tested on
    loss_fn : torch.nn.Module
        A PyTorch loss function to calculate loss on the test data
    device : torch.device
        A target device to compute on (e.g "cuda" or "cpu")

    Returns
    -------
    Tuple[float, float]
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy)
    """
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred_logits = model(X)
             
            # 2. Calcualte and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # 3. Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
            
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    return test_loss, test_acc

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device
) -> Dict[str, List]:
    
    """Trains and tests a PyTorch model.

    Passes a target PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model to be trained and tested.
    train_dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be trained on.
    test_dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be tested on.
    optimizer : torch.optim.Optimizer
        A PyTorch optimizer to help minimize the loss function.
    loss_fn : torch.nn.Module
        A PyTorch loss function to calculate loss on both datasets.
    epochs : int
        An integer indicating how many epochs to train for.
    device : torch.device
        A target device to compute on (e.g. "cuda" or "cpu").

    Returns
    -------
    Dict[str, List]
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        
        In the form:
        {'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []}
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                           dataloader=test_dataloader,
                                           loss_fn=loss_fn,
                                           device=device)
        
        # print status
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"  
        )
        
        # update results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)        
    
    return results
