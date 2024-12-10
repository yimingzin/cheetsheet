import math
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader

def show_image_mnist(images):
    """可视化 - 显示batch_size数量的图像
    sqrtn: 假设batch_size = 128 , 开根号后为11.3,ceil向上取整12,网格数为12x12 = 144，通过空余16个来显示整个batch_size的图像
    sqrtimg: 图片是28x28，通过reshape变为1x28x28=784计算平方根来求得显示时的边长
        
    Args:
        images (_type_): next(iter(train_datalodaer))[0] => images.shape = (batch_size, 1, 28, 28) to MINST
    """
    N, C, H, W = images.shape
    images = torch.reshape(images, (N, -1))   # images reshape to (batch_size, 1x28x28)
    sqrtn = int(math.ceil(math.sqrt(N)))      # 假设batch_size = 128 , 开根号后为11.3,ceil向上取整12,网格数为12x12 = 144，通过空余16个来显示整个batch_size的图像
    sqrtimg = int(math.ceil(math.sqrt(H * W)))    # 图片是28x28，通过reshape变为1x28x28=784计算平方根来求得显示时的边长

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape((sqrtimg, sqrtimg)), cmap="gray")
    
    plt.show()
    return

MNIST_train_data = datasets.MNIST(
    root="data/MNIST_data",
    train=True,
    transform=T.ToTensor(),
    download=True
)

MNIST_train_dataloader = DataLoader(
    dataset=MNIST_train_data,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


imgs, label = next(iter(MNIST_train_dataloader))
show_image_mnist(imgs)