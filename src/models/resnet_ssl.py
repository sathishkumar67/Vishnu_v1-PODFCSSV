import torch
import torch.nn as nn
from torchvision.models import resnet18
from src.models.adapter import IBALocalAdapter

class SimCLR_ResNet18_IBA(nn.Module):
    def __init__(self, output_dim=128, adapter_dim=64):
        super(SimCLR_ResNet18_IBA, self).__init__()
        
        # 1. Load Pre-trained ResNet
        self.backbone = resnet18(pretrained=True)
        
        # 2. Freeze the entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Inject Adapters (IBA-Local)
        # We hook into the layers: layer1, layer2, layer3, layer4
        self.adapter1 = IBALocalAdapter(64, bottleneck_dim=adapter_dim)
        self.adapter2 = IBALocalAdapter(128, bottleneck_dim=adapter_dim)
        self.adapter3 = IBALocalAdapter(256, bottleneck_dim=adapter_dim)
        self.adapter4 = IBALocalAdapter(512, bottleneck_dim=adapter_dim)
        
        # 4. Projection Head (for SSL contrastive loss)
        # Input to head is ResNet18's fc input dim (512)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        # Remove original classification head
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        # Manual forward pass to inject adapters
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1 + Adapter
        x = self.backbone.layer1(x)
        x = self.adapter1(x)

        # Layer 2 + Adapter
        x = self.backbone.layer2(x)
        x = self.adapter2(x)

        # Layer 3 + Adapter
        x = self.backbone.layer3(x)
        x = self.adapter3(x)

        # Layer 4 + Adapter
        x = self.backbone.layer4(x)
        x = self.adapter4(x)

        # Global Pooling
        x = self.backbone.avgpool(x)
        feature_vector = torch.flatten(x, 1) # This is the "Representation"
        
        # Projection Head
        z = self.projection_head(feature_vector)
        
        return feature_vector, z

    def get_trainable_params(self):
        # Helper to optimize only adapters and head
        params = []
        params += list(self.adapter1.parameters())
        params += list(self.adapter2.parameters())
        params += list(self.adapter3.parameters())
        params += list(self.adapter4.parameters())
        params += list(self.projection_head.parameters())
        return params