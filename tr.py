import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class MRIDataset(Dataset):
    def __init__(self, study_ids, labels_df, coords_df, image_dir, transform=None):
        self.study_ids = study_ids
        self.labels_df = labels_df
        self.coords_df = coords_df
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.study_ids)
    
    def load_dicom(self, study_id):
        # Get the first available series for this study
        study_path = os.path.join(self.image_dir, str(study_id))
        series_folders = os.listdir(study_path)
        if len(series_folders) == 0:
            return None
        
        # Get the first instance in the first series
        series_path = os.path.join(study_path, series_folders[0])
        instances = os.listdir(series_path)
        if len(instances) == 0:
            return None
            
        dicom_path = os.path.join(series_path, instances[0])
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        
        # Normalize image
        image = image - image.min()
        image = image / (image.max() + 1e-8)
        image = (image * 255).astype(np.uint8)
        
        # Convert to 3 channels if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
            
        return image
    
    def get_labels(self, study_id):
        study_labels = self.labels_df[self.labels_df['study_id'] == study_id]
        
        # Initialize labels dictionary
        labels = {
            'spinal_canal_stenosis': np.zeros(3),  # [normal/mild, moderate, severe]
            'neural_foraminal_narrowing': np.zeros(3),
            'subarticular_stenosis': np.zeros(3)
        }
        
        # Fill in the labels
        for _, row in study_labels.iterrows():
            for col in row.index:
                if '_l' in col:  # This identifies the condition columns
                    condition = col.split('_l')[0]
                    if condition in labels:
                        severity = row[col]
                        if pd.notna(severity):
                            if severity.lower() in ['normal', 'mild']:
                                labels[condition][0] = 1
                            elif severity.lower() == 'moderate':
                                labels[condition][1] = 1
                            elif severity.lower() == 'severe':
                                labels[condition][2] = 1
        
        return labels

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        
        # Load image
        image = self.load_dicom(study_id)
        if image is None:
            # Handle missing image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get labels
        labels = self.get_labels(study_id)
        
        # Convert labels to tensor
        label_tensor = torch.tensor([
            *labels['spinal_canal_stenosis'],
            *labels['neural_foraminal_narrowing'],
            *labels['subarticular_stenosis']
        ], dtype=torch.float32)
        
        return image, label_tensor

# [Previous imports and MRIDataset class remain the same]

class MRIConditionModel(nn.Module):
    def __init__(self, backbone='resnet18', num_conditions=3):
        super().__init__()
        # Initialize the backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            in_chans=3,  # Explicitly specify 3 input channels
            features_only=False,  # We want the full model
            num_classes=0  # No classification head
        )
        
        # Get the number of features from the backbone
        self.num_features = self.backbone.num_features  # Should be 512 for ResNet18
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_conditions * 3)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone.forward_features(x)  # Use forward_features instead of direct forward
        
        # Apply classifier
        x = self.classifier(features)
        
        return x

# Update the training function with better error handling and monitoring
def train_model(model, train_loader, val_loader, device, epochs=50, patience=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stop = False
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Debug information for first batch
            if batch_idx == 0:
                print(f"\nBatch info:")
                print(f"Image batch shape: {images.shape}")
                print(f"Target batch shape: {targets.shape}")
                print(f"Image dtype: {images.dtype}")
                print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
            
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(images)
                
                # Debug information for first batch
                if batch_idx == 0:
                    print(f"Output shape: {outputs.shape}")
                    print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                train_loss += loss.item()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}:")
                print(f"Exception: {str(e)}")
                print(f"Image shape: {images.shape}")
                print(f"Image dtype: {images.dtype}")
                print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
                raise e
        
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, '/kaggle/working/best_mri_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                early_stop = True
                break
        
        print(f"Epochs without improvement: {epochs_without_improvement}")
        
        if early_stop:
            break
    
    if not early_stop:
        print("\nTraining completed for all epochs!")
    else:
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        # Load best model
        checkpoint = torch.load('/kaggle/working/best_mri_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
def main():
    # Data Preparation
    train_df = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
    label_coords_df = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')
    
    # Get unique study IDs
    study_ids = train_df['study_id'].unique()
    train_ids, val_ids = train_test_split(study_ids, test_size=0.2, random_state=42)
    
    # Create transforms
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    # Create Datasets
    train_dataset = MRIDataset(
        train_ids,
        train_df,
        label_coords_df,
        image_dir='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images',
        transform=transform
    )
    
    val_dataset = MRIDataset(
        val_ids,
        train_df,
        label_coords_df,
        image_dir='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images',
        transform=transform
    )
    
    # Test single item loading
    test_image, test_label = train_dataset[0]
    print(f"Test image shape: {test_image.shape}")
    print(f"Test label shape: {test_label.shape}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=2,
        pin_memory=True
    )
    
    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MRIConditionModel(backbone='resnet18', num_conditions=3).to(device)
    
    print("\nStarting training...")
    print(f"Device being used: {device}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    # Training
    trained_model = train_model(model, train_loader, val_loader, device)
    
    # Save final model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'study_ids': study_ids,
    }, '/kaggle/working/final_mri_model.pth')

if __name__ == '__main__':
    main()
