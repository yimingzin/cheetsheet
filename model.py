import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets
from train import train

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

CIFAR10_train_dataloader = DataLoader(
    CIFAR10_train_data,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

CIFAR10_test_dataloader = DataLoader(
    CIFAR10_test_data,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

epochs = 1
lr = 1e-3
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

single_epoch_results = train(
    epochs=epochs,
    model=model,
    train_dataloader=CIFAR10_train_dataloader,
    test_dataloader=CIFAR10_test_dataloader,
    loss_func=loss_func,
    optimizer=optimizer,
    device=device
)