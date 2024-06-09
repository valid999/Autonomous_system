# We need to install pyt orch library with cuda device to run the code 

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Step 1: Capture Images
def capture_images(num_images=10000000, save_dir='path_to_images'):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)  # Open laptop's camera
    for i in range(num_images):
        ret ,frame = cap.read()
        if not ret :
            print("Failed to capture image from the camera")
            continue
        cv2.imwrite(os.path.join(save_dir, f'image_{i}.jpg'), frame)
        cv2.imshow('Captured Image', frame)
        cv2.waitKey(10000)  # Show the captured image for 100 milliseconds
    cap.release()
    cv2.destroyAllWindows()

# Step 2: Store Images
capture_images(num_images=100000, save_dir='path_to_images')

# Step 3: Prepare Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
])

dataset = CustomDataset(image_dir='path_to_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # We can put different batch size in the system 

# Step 4: Train Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 30 * 40, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 30 * 40)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images in dataloader:
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, torch.randn_like(outputs))  # Random labels for demonstration
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Save the model
torch.save(model.state_dict(), 'simple_cnn_model.pth')