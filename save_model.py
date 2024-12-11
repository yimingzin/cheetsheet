import torch
from pathlib import Path

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
):
    """Saves a Pytorch model to a target airectory

    Args:
        model (torch.nn.Module): A target Pytorch model to save
        target_dir (str): A directory for saving the model to.
        model_name (str): A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.
    """
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    # Save the model state_dict()
    print(f"[INFO] Saving model to : {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
