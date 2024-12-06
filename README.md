# Pneumonia Detector using Computer Vision 

This project uses the DenseNet121 model to detect pneumonia in chest X-ray images. The model is trained on a dataset with train, validation, and test splits from kaggle and demonstrates the use of transfer learning and data augmentation in PyTorch.

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Model Architecture](#model-architecture)
3. [Implementation](#implementation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Future Work](#future-work)

---

## Dataset Overview
The dataset is structured as follows:
```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```
Each folder contains JPEG images of chest X-rays. 
You can download a similar dataset from [Kaggle Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

---

## Model Architecture

The model uses **DenseNet121**, a pre-trained model available in `torchvision.models`. Its final layer is replaced with a fully connected layer for binary classification.

```python
# Load DenseNet121 model
model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)  # Binary classification (NORMAL vs PNEUMONIA)
```

---

## Implementation

Here is the Python implementation of the project:

### Data Preparation and Pre-Processing

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset paths
train_dir = "/training_dataset"
val_dir = "/val_dataset"
test_dir = "/test_dataset"

# Image transformation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([...]),  
    'test': transforms.Compose([...]), 
}

# Load datasets
datasets_dict = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
}

# Create dataloaders
dataloaders = {
    'train': DataLoader(datasets_dict['train'], batch_size=32, shuffle=True),
    'val': DataLoader(datasets_dict['val'], batch_size=32, shuffle=False),
    'test': DataLoader(datasets_dict['test'], batch_size=32, shuffle=False),
}
```

### Training

```python

import time
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_postfix(loss=..., accuracy=...)
```

---

## Training

Run the training script to train the DenseNet121 model for 5 epochs:
```bash
python main.py
```

During training, the script displays a progress bar and updates metrics like loss and accuracy.

---

## Evaluation

The test dataset is used to evaluate the trained model:

```python

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
           for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        accuracy = running_corrects.double() / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Results

### Sample Metrics
- **Validation Accuracy**: ~81%
- **Test Accuracy**: ~88%

### Sample Confusion Matrix
|                | Predicted NORMAL | Predicted PNEUMONIA |
|----------------|------------------|----------------------|
| **Actual NORMAL** | 500              | 50                   |
| **Actual PNEUMONIA** | 30               | 700                  |

---

## Images

### Example Input
#### NORMAL X-Ray:
![Normal Chest X-Ray](https://example.com/normal.jpg)

#### PNEUMONIA X-Ray:
![Pneumonia Chest X-Ray](https://example.com/pneumonia.jpg)

---

## Future Work

1. Hyperparameter optimization for better accuracy.
2. Explore additional augmentation techniques.
3. Deploy the model as a web application using Flask or FastAPI.
4. Experiment with other models like ResNet or EfficientNet.

---

## References

- [DenseNet121 Documentation](https://pytorch.org/vision/stable/models.html)
- [Chest X-Ray Dataset](https://www.kaggle.com/datasets)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)
