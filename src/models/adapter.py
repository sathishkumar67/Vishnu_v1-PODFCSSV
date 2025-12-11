import torch
import torch.nn as nn

class IBALocalAdapter(nn.Module):
    """
    IBA-Local: Information-Bottlenecked Adapter.
    A lightweight bottleneck module injected into frozen backbones.
    Structure: Input -> Down proj -> ReLU -> Up proj -> Output
    """
    def __init__(self, in_channels, bottleneck_dim=64):
        super(IBALocalAdapter, self).__init__()
        self.down_project = nn.Linear(in_channels, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, in_channels)
        
        # Initialize weights near zero to ensure identity-like behavior at start
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        # We process spatially: strictly pixel-wise or global average? 
        # For ResNets, adapters often work on the channel dimension.
        
        identity = x
        b, c, h, w = x.shape
        
        # Reshape to [Batch, H*W, Channels] for Linear layer
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        
        # Bottleneck transformation
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        
        # Reshape back to [Batch, Channels, Height, Width]
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        return identity + x  # Residual connection