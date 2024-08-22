import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import ResNet101_Weights
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm
# Pretrain ResNet101 
net = models.resnet101(weights=None, progress=True)
checkpoint_path = 'path/to/resnet101-cd907fc2.pth'
checkpoint = torch.load(checkpoint_path)

net.load_state_dict(checkpoint)


net.eval()

# Data
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load ImageNet val
imagenet_val_dataset = datasets.ImageFolder(root='path/to/val', transform=preprocess)
val_loader = DataLoader(imagenet_val_dataset, batch_size=32, shuffle=False, num_workers=4)


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if predicted!=labels:
            #     print(total%25)
            #     print(labels)
    accuracy = 100 * correct / total
    return accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

accuracy = evaluate_model(net, val_loader)
print(f'Accuracy of the model on the ImageNet validation set: {accuracy:.2f}%')
