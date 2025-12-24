from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import ViTMAEForPreTraining, PreTrainedModel

class IBA_Adapter(nn.Module):
    """
    Information-Bottlenecked Adapter (IBA) module.

    This module implements a bottleneck architecture (Down-project -> Activation -> Up-project)
    inserted into frozen networks to introduce trainable parameters for adaptation.
    
    Architecture:
        Input [B, L, D] -> Linear(D, d) -> Activation -> Linear(d, D) -> Dropout -> + Residual
    
    Attributes:
        down_project (nn.Linear): Dimensionality reduction layer.
        activation (nn.Module): Non-linear activation function.
        up_project (nn.Linear): Dimensionality restoration layer.
        dropout (nn.Dropout): Regularization layer.
    """

    def __init__(
        self, 
        input_dim: int, 
        bottleneck_dim: int = 64, 
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU()
    ) -> None:
        """
        Initializes the IBA Adapter.

        Args:
            input_dim (int): The hidden dimension of the backbone model (e.g., 768 for ViT-Base).
            bottleneck_dim (int): The reduced dimension for the bottleneck. Lower values 
                                  compress information more (Information Bottleneck principle).
            dropout (float): Dropout probability applied after the up-projection
            activation (nn.Module): Activation function to use between projections.
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation

        # Down-projection: Compress semantic information
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        # Up-projection: Reconstruct features for the next layer
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Weight Initialization Strategy:
        # 1. Kaiming Normal for down_project to maintain variance through the non-linearity.
        # 2. Zeros for up_project. This ensures the adapter acts as an identity function 
        #    at initialization (Adapter(x) = 0), preventing semantic shock to the frozen backbone.
        nn.init.kaiming_normal_(self.down_project.weight)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch_Size, Seq_Len, Hidden_Dim].

        Returns:
            torch.Tensor: Adapted features of the same shape as input.
        """
        residual = x
        
        # Bottleneck compression
        x = self.down_project(x)
        x = self.activation(x)
        
        # TODO(Future): Inject variational noise here for the probabilistic extension (Zeus/V4).
        # x = x + torch.randn_like(x) * self.noise_scale 
        
        # Reconstruction & Regularization
        x = self.up_project(x)
        x = self.dropout(x)
        
        # Residual connection preserves original features while adding adaptation
        return residual + x

    def __repr__(self):
        """Custom string representation for easier debugging."""
        return f"IBA_Adapter(in={self.input_dim}, btl={self.bottleneck_dim})"


class ViTBlockWithAdapter(nn.Module):
    """
    Wrapper class to inject an Adapter into a Hugging Face ViTLayer.

    It intercepts the output of the original frozen block, passes the hidden states
    through the adapter, and repackages the output to match Hugging Face's return signature.
    """

    def __init__(self, original_block: nn.Module, adapter: IBA_Adapter) -> None:
        """
        Args:
            original_block (nn.Module): The original, frozen Transformer block.
            adapter (IBA_Adapter): The trainable adapter instance.
        """
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        head_mask: Optional[torch.Tensor] = None, 
        output_attentions: bool = False,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass matching standard Hugging Face ViTLayer signature.
        """
        # 1. Run the original frozen ViT Block
        # HF blocks return a tuple: (hidden_states, attention_weights (optional), ...)
        outputs = self.original_block(
            hidden_states, 
            head_mask=head_mask, 
            output_attentions=output_attentions,
            **kwargs
        )
        
        # 2. Extract Hidden States (Always the first element)
        x = outputs[0]
        
        # 3. Apply the IBA Adapter
        x = self.adapter(x)
        
        # 4. Repackage output to maintain compatibility with HF pipeline
        if output_attentions:
            return (x,) + outputs[1:]
        
        return (x,)


def inject_adapters(model: PreTrainedModel, bottleneck_dim: int = 64) -> PreTrainedModel:
    """
    Injects IBA Adapters into the Encoder of a ViTMAE model.

    This function performs the following operations:
    1. Freezes all existing parameters in the model.
    2. Identifies the Encoder layers.
    3. Wraps each layer with `ViTBlockWithAdapter`.
    4. Unfreezes ONLY the new Adapter parameters.

    Args:
        model (PreTrainedModel): The Hugging Face ViTMAE model instance.
        bottleneck_dim (int): Dimension of the adapter bottleneck.

    Returns:
        PreTrainedModel: The modified model with adapters injected.
    """
    print(f"\n{'='*60}")
    print(f"[System] Starting Adapter Injection Procedure")
    print(f"{'='*60}")

    # 1. Freeze the entire model backbone
    print("[Config] Freezing original backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Locate the Encoder
    # Safety check: Ensure the model structure is what we expect
    if not hasattr(model, "vit") or not hasattr(model.vit, "encoder"):
        raise AttributeError("The provided model does not have a standard 'vit.encoder' attribute.")

    encoder = model.vit.encoder
    input_dim = model.config.hidden_size
    num_layers = len(encoder.layer)

    print(f"[Config] Model Config: Hidden Dim={input_dim}, Layers={num_layers}")
    print(f"[Config] Adapter Config: Bottleneck Dim={bottleneck_dim}")

    # 3. Iterate and Replace
    print("[Action] Injecting adapters into encoder layers...")
    
    for i, layer in enumerate(encoder.layer):
        # Instantiate the adapter
        adapter = IBA_Adapter(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
        
        # Ensure adapter is on the same device and dtype as the layer it wraps
        # This handles cases where the model is already on GPU or in FP16
        ref_param = next(layer.parameters())
        adapter.to(device=ref_param.device, dtype=ref_param.dtype)
        
        # Wrap the original layer
        wrapped_layer = ViTBlockWithAdapter(original_block=layer, adapter=adapter)
        
        # Mutate the ModuleList in-place
        encoder.layer[i] = wrapped_layer
        
        # Simple progress indicator for large models
        if (i + 1) % 4 == 0 or (i + 1) == num_layers:
            print(f"  -> Processed layer {i + 1}/{num_layers}")

    print(f"[System] Injection Complete. Decoder layers ignored (if present).")
    
    # 4. Verification of Trainable Parameters
    count_trainable_params(model)
    
    return model


def count_trainable_params(model: nn.Module) -> None:
    """Utility to print the count of frozen vs trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    ratio = (trainable_params / total_params) * 100
    
    print(f"\n[Stats] Parameter Audit:")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Frozen Backbone:      {frozen_params:,}")
    print(f"  - Trainable (Adapters): {trainable_params:,}")
    print(f"  - Trainable Ratio:      {ratio:.2f}%")
    print(f"{'='*60}\n")

# # =============================================================================
# # Main Execution Block (For Testing)
# # =============================================================================
# if __name__ == "__main__":
#     # Simulate loading a model
#     print("[Main] Loading pre-trained ViTMAE...")
#     try:
#         model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        
#         # Inject Adapters
#         model = inject_adapters(model, bottleneck_dim=64)
        
#         # Sanity Check: Forward pass
#         print("[Main] Running dummy forward pass to verify graph integrity...")
#         dummy_input = torch.randn(1, 3, 224, 224)
        
#         # Forward pass (ensure gradients flow through adapters)
#         output = model(dummy_input)
        
#         print(f"[Success] Forward pass complete. Loss: {output.loss.item():.4f}")
        
#     except Exception as e:
#         print(f"[Error] An error occurred during execution: {e}")