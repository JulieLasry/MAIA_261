import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ChestXRayModel(nn.Module):
    """
    Model class using a standard ResNet-50 backbone from torchvision
    pretrained on ImageNet, trained in two phases. Dropout is added 
    here in the ResNet classification head for regularization, as 
    Global Average Pooling and Batch Normalization are already part 
    of the model architecture. 
    Two phases are implemented. The phase 1 is a feature extraction phase,
    where the backbone weights are frozen and only the classification
    head is trained, leveraging the pretrained ImageNet features.
    Phase 2, triggered externally via unfreeze_last_stages(), 
    progressively unfreezes the last backbone stages for finetuning,
    with a lower learning rate handled by the training loop.

    Args:
        training_config (dict): The dict of the training main
        paths and variable values in training process.
    """
    def __init__(
        self,
        training_config: dict,
    ) -> None:
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        num_features: int = self.backbone.fc.in_features  # 2048 output features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=training_config['hyperparameters']['dropout']),
            nn.Linear(num_features, training_config['model']['num_classes'])
        )

        # Phase 1 - Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Train only the classification head weights
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_last_stages(
        self,
        num_stages: int,
    ) -> None:
        """
        Progressively unfreezes only the last residual 
        layer groups of the backbone (from layer 4 to max layer 1 
        for a global finetuning).

        Args:
            num_stages (int): Number of residual layers to
            unfreeze in phase 2 (max 4 : layer1 to layer4).
        """
        resnet_layers = [
            self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4
        ]
        for layer in resnet_layers[-num_stages:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone, including the
        classification head.

        Args:
            x (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor
            The raw classification logits for each image.
        """
        return self.backbone(x)

    def return_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent embedding of the image, extracted just
        before the final classification (fc) layer right after Global 
        Average Pooling, to be used by the latent space visualization.

        Args:
            x (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor
            The 1D vector of the features embedding.  
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1) # to have a vector
        return x
