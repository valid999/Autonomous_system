# The code working with Python3.10 + the environment details going
# to be in the GitHub: https://github.com/valid999/Final_Project

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms# pip install torchvision if not availabel
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F


# Define a custom dataset class
class SelfDrivingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg') or fname.endswith('.png')])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# Define data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Load dataset and split into training and validation sets
img_dir = 'path_to_images' # The path should bed the same name 
dataset = SelfDrivingDataset(img_dir=img_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                            [train_size, val_size])


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


# Define the convolutional autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec5 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):
        # Encoder
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        # Decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = torch.sigmoid(self.dec5(x))
        return x


# Explanation of the  64 channels the dapth  i will provide more information in more details in the project 
"""
Here's why we have 64 channels in the last layer:

Channels in the Last Decoder Layer (64):
The number of channels in the last decoder layer depends on the desired complexity of the
reconstructed image.
In this case, 64 channels might have been chosen based on the complexity of the input images
and the depth of the encoder layers.
This number can be adjusted based on factors such as the complexity of the data, the depth of 
the network, and computational resources.
"""


# The channels of the color must be as the same input they have taken and the majority of it is 3 becasue of the RGB
"""

Channels Matching the Input Image:
The number of channels in the last decoder layer should match the number of channels in the input image.
For RGB images, there are typically 3 channels representing the red, green, and blue color channels.
Therefore, the output of the last decoder layer should also have 3 channels to reconstruct the 
color information.

"""




# Instantiate the model and move it to GPU if available
model = ConvAutoencoder().cuda()


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.cuda()

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images in val_loader:
            images = images.cuda()
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(images.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Calculate evaluation metrics
    mse = np.mean((np.array(all_predictions) - np.array(all_labels))**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_labels)))

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.6f}, "
          f"Val")

