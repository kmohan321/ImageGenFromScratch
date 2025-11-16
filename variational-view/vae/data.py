from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from config import config

batch_size = config['batch_size']

train_data = MNIST(f"data",train=True,download=True,transform=transforms.ToTensor())
test_data = MNIST(f"data",train=True,download=True,transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
