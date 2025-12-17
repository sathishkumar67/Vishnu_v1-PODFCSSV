import torch
import torch.nn as nn
import timm
import numpy as np

class MAE_ViT(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224', mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # 1. Load Pretrained Encoder (ViT)
        # We remove the classifier head (num_classes=0) because we only want features
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # Extract specs from encoder
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.num_patches = self.encoder.patch_embed.num_patches
        
        # 2. Lightweight Decoder
        # A small Transformer to reconstruct the image from features
        decoder_dim = 128
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_dim)
        
        # Learnable mask token (replaces missing patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Positional embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim), requires_grad=False
        )
        
        # Tiny transformer blocks for decoder (fewer layers than encoder for efficiency)
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=4, dim_feedforward=512, batch_first=True)
            for _ in range(4)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        
        # 3. Prediction Head
        # Projects back to pixel space: (Patch_Size * Patch_Size * 3 colors)
        pixels_per_patch = self.patch_size * self.patch_size * 3
        self.decoder_pred = nn.Linear(decoder_dim, pixels_per_patch)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Sine-cosine positional embeddings logic could go here
        # For MVP, we initialize mask token and linear layers
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.xavier_uniform_(self.decoder_pred.weight)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by shuffling keys
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # Batch, Length, Dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # Noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask in the original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # 1. Patch Embed
        x = self.encoder.patch_embed(x)
        
        # 2. Add Positional Embeddings (Standard ViT logic)
        # We discard the CLS token pos embed for masking simplicity in this MVP
        cls_pos_embed = self.encoder.pos_embed[:, 0, :]
        patch_pos_embed = self.encoder.pos_embed[:, 1:, :]
        x = x + patch_pos_embed
        
        # 3. Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 4. Append CLS token
        cls_token = self.encoder.cls_token + cls_pos_embed
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 5. Apply Transformer Blocks
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # 1. Embed to decoder dimension
        x = self.decoder_embed(x)

        # 2. Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token back

        # 3. Add decoder pos embed
        # Interpolate if needed, but for fixed size we just slice
        x = x + self.decoder_pos_embed[:, :x.shape[1], :]

        # 4. Apply Transformer Decoder
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 5. Predict pixels
        x = self.decoder_pred(x)

        # Remove CLS token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        
        # MSE Loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # Only compute loss on masked patches (MIM principle)
        loss = (loss * mask).sum() / mask.sum() 
        return loss

    def patchify(self, imgs):
        """
        img: (N, 3, H, W) -> (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask