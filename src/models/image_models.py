"""
Image model implementations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel


class SimpleCNN(nn.Module):
    """Simple CNN architecture for image classification"""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNNModel(BaseModel):
    """CNN model for image data"""
    
    supported_data_types = ['image']
    
    def _create_model(self) -> SimpleCNN:
        """Create CNN model"""
        dataset_info = self.dataset.get_info()
        num_classes = dataset_info['n_classes']
        image_size = dataset_info.get('image_size', (224, 224))
        
        # Determine input channels based on image shape
        # MNIST is grayscale (1 channel), others are RGB (3 channels)
        if len(image_size) == 2:
            # Check if this is likely grayscale (square images, small size)
            if image_size[0] <= 32 and image_size[1] <= 32:
                input_channels = 1  # Likely grayscale like MNIST
            else:
                input_channels = 3  # Likely RGB
        else:
            input_channels = 3  # Default to RGB
        
        return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the CNN model"""
        # Convert numpy arrays to tensors
        # Handle different input formats: grayscale vs RGB
        if X_train.ndim == 4:  # (batch, channels, height, width)
            if X_train.shape[1] == 1:  # Grayscale
                X_train_tensor = torch.FloatTensor(X_train)
            else:  # RGB
                X_train_tensor = torch.FloatTensor(X_train)
        elif X_train.ndim == 3:  # (batch, height, width) - grayscale
            X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        else:  # (batch, height, width, channels)
            if X_train.shape[-1] == 1:  # Grayscale (H, W, 1) -> (1, H, W)
                X_train_tensor = torch.FloatTensor(X_train).squeeze(-1).unsqueeze(1)
            else:  # RGB (H, W, 3) -> (3, H, W)
                X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(10):  # Simple training for now
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        # Handle different input formats: grayscale vs RGB
        if X.ndim == 4:  # (batch, channels, height, width)
            if X.shape[1] == 1:  # Grayscale
                X_tensor = torch.FloatTensor(X)
            else:  # RGB
                X_tensor = torch.FloatTensor(X)
        elif X.ndim == 3:  # (batch, height, width) - grayscale
            X_tensor = torch.FloatTensor(X).unsqueeze(1)
        else:  # (batch, height, width, channels)
            if X.shape[-1] == 1:  # Grayscale (H, W, 1) -> (1, H, W)
                X_tensor = torch.FloatTensor(X).squeeze(-1).unsqueeze(1)
            else:  # RGB (H, W, 3) -> (3, H, W)
                X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        self.model.eval()
        # Handle different input formats: grayscale vs RGB
        if X.ndim == 4:  # (batch, channels, height, width)
            if X.shape[1] == 1:  # Grayscale
                X_tensor = torch.FloatTensor(X)
            else:  # RGB
                X_tensor = torch.FloatTensor(X)
        elif X.ndim == 3:  # (batch, height, width) - grayscale
            X_tensor = torch.FloatTensor(X).unsqueeze(1)
        else:  # (batch, height, width, channels)
            if X.shape[-1] == 1:  # Grayscale (H, W, 1) -> (1, H, W)
                X_tensor = torch.FloatTensor(X).squeeze(-1).unsqueeze(1)
            else:  # RGB (H, W, 3) -> (3, H, W)
                X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.numpy()


class SimpleViT(nn.Module):
    """Simple Vision Transformer architecture"""
    
    def __init__(self, num_classes: int, image_size: int = 224, patch_size: int = 16):
        super(SimpleViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_dim, 256)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, 256))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512),
            num_layers=6
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Reshape to patches
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_patches, self.patch_dim)
        
        # Patch embedding
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        x = self.dropout(x)
        
        # Transformer
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        return x


class ViTModel(BaseModel):
    """Vision Transformer model for image data"""
    
    supported_data_types = ['image']
    
    def _create_model(self) -> SimpleViT:
        """Create ViT model"""
        dataset_info = self.dataset.get_info()
        num_classes = dataset_info['n_classes']
        image_size = dataset_info['image_size'][0]  # Assuming square images
        
        return SimpleViT(num_classes=num_classes, image_size=image_size)
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the ViT model"""
        # Convert numpy arrays to tensors
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        # Training loop
        self.model.train()
        for epoch in range(5):  # Simple training for now
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.numpy() 