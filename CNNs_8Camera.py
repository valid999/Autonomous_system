import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

class MultiCameraDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dirs[0]))

    def __getitem__(self, idx): # enabling the Python objects to behave like sequences or containers e.g lists, dictionaries, and tuples
        images = []
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, f'image_{idx}.jpg')
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return torch.stack(images)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_dirs = ['path_to_images_camera1', 'path_to_images_camera2', ..., 'path_to_images_camera8']
dataset = MultiCameraDataset(img_dirs=img_dirs, transform=transform)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

class MultiCameraCNN(nn.Module):
    def __init__(self):
        super(MultiCameraCNN, self).__init__()
        self.conv1 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 512 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = MultiCameraCNN().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.cuda()

        optimizer.zero_grad()

        outputs = model(images)
        # No labels for unsupervised learning
        # Skip loss calculation

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), 'multi_camera_model.pth')
