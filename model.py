import torch
import torchvision
import argparse
from plot.plot_loss_curves import plot_loss_curves
from save_model import save_model
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets
from train import train

def get_args():
    parser = argparse.ArgumentParser(description="Train the pretrained Neural Network on images classifier task.")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=2, help="Number of epochs")
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-3, help='learning rate', dest='lr')
    
    return parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights).to(device=device)
transform = weights.transforms()

print(transform)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(
    nn.Dropout(p = 0.2, inplace=True),
    nn.Linear(in_features=2048, out_features=10)
).to(device)


CIFAR10_train_data = datasets.CIFAR10(
    root="data/CIFAR10_data",
    train=True,
    transform=transform,
    download=True
)

CIFAR10_test_data = datasets.CIFAR10(
    root="data/CIFAR10_data",
    train=False,
    transform=transform,
    download=True
)

args = get_args()

CIFAR10_train_dataloader = DataLoader(
    CIFAR10_train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0
)

CIFAR10_test_dataloader = DataLoader(
    CIFAR10_test_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0
)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

single_epoch_results = train(
    epochs=args.epochs,
    model=model,
    train_dataloader=CIFAR10_train_dataloader,
    test_dataloader=CIFAR10_test_dataloader,
    loss_func=loss_func,
    optimizer=optimizer,
    device=device
)

plot_loss_curves(single_epoch_results)
save_model(model=model, target_dir="models", model_name="pretrained_Resnet50_CIFAR10_model.pth")