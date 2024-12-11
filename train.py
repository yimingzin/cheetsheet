import time
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Tuple, List

def train_step(
    current_epoch: int,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    disable_progress_bar: bool = False
) -> Tuple[float, float]:
    """single epoch for train 

    Args:
        current_epoch (int): _description_
        model (torch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        loss_func (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        device (torch.device): _description_
        disable_progress_bar (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[float, float]: _description_
    """
    
    model.train()
    train_loss, train_acc = 0, 0
    
    progress_bar = tqdm(
        iterable=enumerate(train_dataloader),
        desc=f"Training Epoch {current_epoch}",
        total=len(train_dataloader),
        disable=disable_progress_bar
    )
    
    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        
        train_logits = model(X)
        loss = loss_func(train_logits, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        train_probs = torch.softmax(train_logits, dim=1)
        train_preds = torch.argmax(train_probs, dim=1)
        
        train_acc += (train_preds == y).sum().item() / len(train_logits)
        
        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
                "train_acc": train_acc / (batch + 1)
            }
        )
    
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    
    return train_loss, train_acc


def test_step(
    current_epoch: int,
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    device: torch.device,
    disable_progress_bar: bool = False
) -> Tuple[float, float]:
    """ Test single epoch in Test

    Args:
        current_epoch (int): _description_
        model (torch.nn.Module): _description_
        test_dataloader (torch.utils.data.DataLoader): _description_
        loss_func (torch.nn.Module): _description_
        device (torch.device): _description_
        disable_progress_bar (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[float, float]: _description_
    """
    
    model.eval()
    test_loss, test_acc = 0, 0
    
    progress_bar = tqdm(
        iterable=enumerate(test_dataloader),
        desc=f"Testing Epoch {current_epoch}",
        total=len(test_dataloader),
        disable=disable_progress_bar
    )
    
    # with torch.inference_mode()
    with torch.no_grad():
        for batch, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)
            test_logits = model(X)
            loss = loss_func(test_logits, y)
            test_loss += loss.item()
            
            test_probs = torch.softmax(test_logits, dim=1)
            test_preds = torch.argmax(test_probs, dim=1)
            
            test_acc += (test_preds == y).sum().item() / len(test_logits)
            
            progress_bar.set_postfix(
                {
                    "test_loss": test_loss / (batch + 1),
                    "test_acc": test_acc / (batch + 1)
                }
            )
        
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    
    return test_loss, test_acc

def train(
    epochs: int,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    disable_progress_bar: bool = False
) -> Dict[str, List]:
    """summary function

    Args:
        epochs (int): _description_
        model (torch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        test_dataloader (torch.utils.data.DataLoader): _description_
        loss_func (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        device (torch.device): _description_
        disable_progress_bar (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[str, list]: _description_
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "train_epoch_time": [],
        "test_epoch_time": []
    }
    
    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        current_epoch = epoch + 1
        
        train_epoch_start_time = time.time()
        train_loss, train_acc = train_step(current_epoch, model, train_dataloader, loss_func, optimizer, device, disable_progress_bar)
        train_epoch_end_time = time.time()
        
        train_epoch_time = train_epoch_end_time - train_epoch_start_time

        test_epoch_start_time = time.time()
        test_loss, test_acc = test_step(current_epoch, model, test_dataloader, loss_func, device, disable_progress_bar)
        test_epoch_end_time = time.time()
        
        test_epoch_time = test_epoch_end_time - test_epoch_start_time
        
        print(
            f"Epoch: {epoch + 1} |"
            f"train_loss: {train_loss:.4f} |"
            f"train_acc: {train_acc:.4f} |"
            f"test_loss: {test_loss:.4f} |"
            f"test_acc: {test_acc:.4f} |"
            f"train_epoch_time: {train_epoch_time:.5f} |"
            f"test_epoch_time: {test_epoch_time:.4f} |"
        )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_epoch_time"].append(train_epoch_time)
        results["test_epoch_time"].append(test_epoch_time)
    
    return results