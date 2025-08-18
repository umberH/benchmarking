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
    
    def __init__(self, num_classes: int, input_channels: int = 3, input_size: tuple = (28, 28)):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        # Dynamically compute the flattened feature size for fc1
        self._feature_dim = self._get_flattened_size(input_channels, input_size)
        self.fc1 = nn.Linear(self._feature_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_flattened_size(self, input_channels, input_size):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            x = self.pool(self.relu(self.conv1(dummy)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNNModel(BaseModel):
    supported_data_types = ['image']
    """CNN model for image data"""
    
    supported_data_types = ['image']
    
    def _create_model(self) -> SimpleCNN:
        """Create CNN model with correct input channels (1 for grayscale, 3 for RGB) and dynamic input size"""
        dataset_info = self.dataset.get_info()
        num_classes = dataset_info['n_classes']
        image_size = dataset_info.get('image_size', (28, 28))
        input_channels = 3  # Default to RGB
        # Try to get a sample to check channel count and size
        try:
            X_train, _, _, _ = self.dataset.get_data()
            # X_train shape: (N, C, H, W) or (N, H, W) or (N, H, W, C)
            if X_train.ndim == 4:
                # (N, C, H, W) or (N, H, W, C)
                if X_train.shape[1] == 1:
                    input_channels = 1
                elif X_train.shape[-1] == 1:
                    input_channels = 1
                # Get H, W
                if X_train.shape[1] == 1 or X_train.shape[1] == 3:
                    image_size = (X_train.shape[2], X_train.shape[3])
                else:
                    image_size = (X_train.shape[1], X_train.shape[2])
            elif X_train.ndim == 3:
                # (N, H, W) - grayscale
                input_channels = 1
                image_size = (X_train.shape[1], X_train.shape[2])
        except Exception:
            pass
        return SimpleCNN(num_classes=num_classes, input_channels=input_channels, input_size=image_size)
    
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
                if X_train.ndim == 4:
                    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
                else:
                    X_train_tensor = torch.FloatTensor(X_train)
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
        
        # Ensure input is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Handle different input formats: grayscale vs RGB
        if X.ndim == 2:  # Flattened data - need to reshape back to image
            # Infer image dimensions from dataset info or make assumptions
            try:
                dataset_info = self.dataset.get_info()
                image_size = dataset_info.get('image_size', (28, 28))
                n_channels = dataset_info.get('n_channels', 1)
                X_reshaped = X.reshape(X.shape[0], n_channels, *image_size)
                X_tensor = torch.FloatTensor(X_reshaped)
            except:
                # Fallback: assume 28x28 grayscale (MNIST-like)
                side_len = int(np.sqrt(X.shape[1]))
                if side_len * side_len == X.shape[1]:
                    X_reshaped = X.reshape(X.shape[0], 1, side_len, side_len)
                else:
                    # Try common sizes
                    if X.shape[1] == 784:  # 28x28
                        X_reshaped = X.reshape(X.shape[0], 1, 28, 28)
                    elif X.shape[1] == 3072:  # 32x32x3 (CIFAR-10)
                        X_reshaped = X.reshape(X.shape[0], 3, 32, 32)
                    else:
                        raise ValueError(f"Cannot infer image shape from flattened data with {X.shape[1]} features")
                X_tensor = torch.FloatTensor(X_reshaped)
        elif X.ndim == 4:  # (batch, channels, height, width)
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
                if X.ndim == 4:
                    X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
                else:
                    X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        self.model.eval()
        
        # Ensure input is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Handle different input formats: grayscale vs RGB
        if X.ndim == 2:  # Flattened data - need to reshape back to image
            # Infer image dimensions from dataset info or make assumptions
            try:
                dataset_info = self.dataset.get_info()
                image_size = dataset_info.get('image_size', (28, 28))
                n_channels = dataset_info.get('n_channels', 1)
                X_reshaped = X.reshape(X.shape[0], n_channels, *image_size)
                X_tensor = torch.FloatTensor(X_reshaped)
            except:
                # Fallback: assume 28x28 grayscale (MNIST-like)
                side_len = int(np.sqrt(X.shape[1]))
                if side_len * side_len == X.shape[1]:
                    X_reshaped = X.reshape(X.shape[0], 1, side_len, side_len)
                else:
                    # Try common sizes
                    if X.shape[1] == 784:  # 28x28
                        X_reshaped = X.reshape(X.shape[0], 1, 28, 28)
                    elif X.shape[1] == 3072:  # 32x32x3 (CIFAR-10)
                        X_reshaped = X.reshape(X.shape[0], 3, 32, 32)
                    else:
                        raise ValueError(f"Cannot infer image shape from flattened data with {X.shape[1]} features")
                X_tensor = torch.FloatTensor(X_reshaped)
        elif X.ndim == 4:  # (batch, channels, height, width)
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
                if X.ndim == 4:
                    X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
                else:
                    X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.numpy()


class SimpleViT(nn.Module):
    """Simple Vision Transformer architecture"""
    
    def __init__(self, num_classes: int, image_size: int = 224, patch_size: int = 16, in_channels: int = 3):
        super(SimpleViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        # Patch embedding will be created dynamically in forward
        self.patch_embedding = None
        # Position embedding will be initialized in forward
        self.position_embedding = None
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512),
            num_layers=6
        )
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size
        in_channels = channels
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches = num_patches_h * num_patches_w
        patch_dim = patch_size * patch_size * in_channels
        
        # Create patch embedding layer if not exists or dimensions changed
        if self.patch_embedding is None or self.patch_embedding.in_features != patch_dim:
            device = x.device
            self.patch_embedding = nn.Linear(patch_dim, 256).to(device)
        
        # Extract patches
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, in_channels, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.permute(0,2,3,1,4,5).contiguous().view(batch_size, num_patches, patch_dim)
        # Patch embedding
        x_patched = self.patch_embedding(patches)
        # Position embedding (initialize if needed)
        if self.position_embedding is None or self.position_embedding.shape[1] != num_patches:
            device = x_patched.device
            self.position_embedding = nn.Parameter(torch.randn(1, num_patches, 256, device=device))
        x_patched = x_patched + self.position_embedding
        x_patched = self.dropout(x_patched)
        # Transformer
        x_patched = x_patched.permute(1, 0, 2)  # (seq_len, batch, features)
        x_patched = self.transformer(x_patched)
        x_patched = x_patched.permute(1, 0, 2)  # (batch, seq_len, features)
        # Global average pooling
        x_patched = x_patched.mean(dim=1)
        # Classification
        x_patched = self.classifier(x_patched)
        return x_patched


class ViTModel(BaseModel):
    supported_data_types = ['image']
    """Vision Transformer model for image data"""
    
    supported_data_types = ['image']
    
    def _create_model(self) -> SimpleViT:
        """Create ViT model"""
        dataset_info = self.dataset.get_info()
        num_classes = dataset_info['n_classes']
        image_size = dataset_info['image_size'][0]  # Assuming square images
        in_channels = dataset_info.get('n_channels', 3)
        return SimpleViT(num_classes=num_classes, image_size=image_size, in_channels=in_channels)
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the ViT model"""
        # Convert numpy arrays to tensors
        if X_train.ndim == 4 and X_train.shape[-1] in [1, 3]:
            X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
        else:
            X_train_tensor = torch.FloatTensor(X_train)
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
        
        # Ensure input is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim == 2:  # Flattened data - need to reshape back to image
            # Infer image dimensions from dataset info or make assumptions
            try:
                dataset_info = self.dataset.get_info()
                image_size = dataset_info.get('image_size', (224, 224))
                n_channels = dataset_info.get('n_channels', 3)
                X_reshaped = X.reshape(X.shape[0], *image_size, n_channels)
                X_tensor = torch.FloatTensor(X_reshaped).permute(0, 3, 1, 2)
            except:
                # Fallback: assume square images
                side_len = int(np.sqrt(X.shape[1] // 3))  # Assume RGB
                if side_len * side_len * 3 == X.shape[1]:
                    X_reshaped = X.reshape(X.shape[0], side_len, side_len, 3)
                    X_tensor = torch.FloatTensor(X_reshaped).permute(0, 3, 1, 2)
                elif side_len * side_len == X.shape[1]:  # Grayscale
                    X_reshaped = X.reshape(X.shape[0], side_len, side_len, 1)
                    X_tensor = torch.FloatTensor(X_reshaped).permute(0, 3, 1, 2)
                else:
                    raise ValueError(f"Cannot infer image shape from flattened data with {X.shape[1]} features")
        elif X.ndim == 4 and X.shape[-1] in [1, 3]:
            X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
        else:
            X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        self.model.eval()
        
        # Ensure input is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim == 2:  # Flattened data - need to reshape back to image
            # Infer image dimensions from dataset info or make assumptions
            try:
                dataset_info = self.dataset.get_info()
                image_size = dataset_info.get('image_size', (224, 224))
                n_channels = dataset_info.get('n_channels', 3)
                X_reshaped = X.reshape(X.shape[0], *image_size, n_channels)
                X_tensor = torch.FloatTensor(X_reshaped).permute(0, 3, 1, 2)
            except:
                # Fallback: assume square images
                side_len = int(np.sqrt(X.shape[1] // 3))  # Assume RGB
                if side_len * side_len * 3 == X.shape[1]:
                    X_reshaped = X.reshape(X.shape[0], side_len, side_len, 3)
                    X_tensor = torch.FloatTensor(X_reshaped).permute(0, 3, 1, 2)
                elif side_len * side_len == X.shape[1]:  # Grayscale
                    X_reshaped = X.reshape(X.shape[0], side_len, side_len, 1)
                    X_tensor = torch.FloatTensor(X_reshaped).permute(0, 3, 1, 2)
                else:
                    raise ValueError(f"Cannot infer image shape from flattened data with {X.shape[1]} features")
        elif X.ndim == 4 and X.shape[-1] in [1, 3]:
            X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
        else:
            X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.numpy() 