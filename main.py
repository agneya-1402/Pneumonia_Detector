import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    device = torch.device("cpu")

    # Dataset paths 
    train_dir = "chest_xray/chest_xray/train"
    val_dir = "chest_xray/chest_xray/val"
    test_dir = "chest_xray/chest_xray/test"

    # Image transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # load datasets
    datasets_dict = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    # data loaders
    dataloaders = {
        'train': DataLoader(datasets_dict['train'], batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(datasets_dict['val'], batch_size=32, shuffle=False, num_workers=4),
        'test': DataLoader(datasets_dict['test'], batch_size=32, shuffle=False, num_workers=4),
    }

    # DenseNet121
    model = models.densenet121(pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)  # Binary classification (NORMAL vs PNEUMONIA)
    model = model.to(device)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train & val function
    def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 20)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # train mode
                else:
                    model.eval()   # eval mode

                running_loss = 0.0
                running_corrects = 0
                total_samples = 0
                start_time = time.time()
                with tqdm(dataloaders[phase], unit="batch") as tepoch:
                    for inputs, labels in tepoch:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()

                        # FW
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # BW
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # stats
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        total_samples += inputs.size(0)

                        tepoch.set_postfix(
                            loss=(running_loss / total_samples), 
                            accuracy=(running_corrects.double().item() / total_samples)
                        )

                epoch_time = time.time() - start_time
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                print(f"Epoch time: {epoch_time:.2f}s")

        return model

    # Train 
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as 'pneumonia_densenet121.pth'.")

    # eval 
    def evaluate_model(model, dataloader):
        model.eval()
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        accuracy = running_corrects.double() / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

    evaluate_model(model, dataloaders['test'])

if __name__ == "__main__":
    main()
