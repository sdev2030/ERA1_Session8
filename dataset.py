# Imports
from torchvision import datasets, transforms
import numpy as np
import torch

# Calculate mean and standard deviation for training dataset
train_data = datasets.CIFAR10('./data', download=True, train=True)

# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
print(x.shape)
# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x, axis=(0, 1))/255
train_std = np.std(x, axis=(0, 1))/255
# print the mean and std
print(train_mean, train_std)

# train and test transforms
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])

test_transforms = transforms.Compose([
#     transforms.ColorJitter([brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1])
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])


train_data = datasets.CIFAR10('./data', download=True, train=True, transform=train_transforms)
test_data = datasets.CIFAR10('./data', download=True, train=False, transform=test_transforms)

SEED = 1

cuda = torch.cuda.is_available()
print('cuda available?', cuda)

torch.manual_seed(SEED)

dataloader_args = dict(shuffle=True, batch_size=128,num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64) 


train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
