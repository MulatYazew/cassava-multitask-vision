"""
Model definitions for AgroVision project.
Contains neural network architectures for Cassava leaf disease classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class CassavaClassifier(nn.Module):
    """
    Neural network classifier for Cassava leaf disease classification.
    Uses EfficientNet backbone with custom classification head.
    """
    
    def __init__(self, num_classes=5, pretrained=True, model_name="efficientnet_b0"):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            model_name (str): Name of the model architecture
        """
        super(CassavaClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        elif model_name == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Replace the original classifier
        if model_name == "efficientnet_b0":
            self.backbone.classifier = self.classifier
        elif model_name == "resnet50":
            self.backbone.fc = self.classifier
        elif model_name == "densenet121":
            self.backbone.classifier = self.classifier
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(num_classes=5, pretrained=True, model_name="efficientnet_b0"):
    """
    Factory function to create a model instance.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        model_name (str): Name of the model architecture
        
    Returns:
        nn.Module: Model instance
    """
    model = CassavaClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name
    )
    return model
