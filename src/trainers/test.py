import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.simple_cnn import SimpleCNN

#Data preprocessing and loading for testing
model = SimpleCNN()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalie((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))
    ])

# testing
correct, total = 0, 0 

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


