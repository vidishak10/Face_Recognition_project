<<<<<<< HEAD
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import numpy as np

# Define dataset directory
data_dir = 'data'
# Initialize Inception Resnet V1 model
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(os.listdir(data_dir))).cuda()

# Data transformations
data_transforms = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
class_names = dataset.classes
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Save the trained model
torch.save(model.state_dict(), 'models/face_recognition_model.pth')

# Save class names
with open('models/class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
=======
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import numpy as np

# Define dataset directory
data_dir = 'data'
# Initialize Inception Resnet V1 model
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(os.listdir(data_dir))).cuda()

# Data transformations
data_transforms = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
class_names = dataset.classes
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Save the trained model
torch.save(model.state_dict(), 'models/face_recognition_model.pth')

# Save class names
with open('models/class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
>>>>>>> origin/main
