import torch
import json
import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FederatedCIFAR100(Dataset):
    """
    Reads the federated_splits.json and loads specific indices for a client/task.
    """
    def __init__(self, root, split_file, client_id, task_id, train=True, transform=None):
        self.root = root
        self.transform = transform
        
        # Load the base CIFAR-100 dataset to access data by index
        # We set download=True to ensure it exists, but we won't use its internal split directly
        self.base_dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True
        )
        
        # Load split configuration
        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        # Get indices for this specific client and task
        # JSON keys are always strings, so we convert IDs to string
        if train:
            self.indices = splits['client_data'][str(client_id)][str(task_id)]
        else:
            # For testing, we use the global test set for this task
            self.indices = splits['test_data'][str(task_id)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map logical index to actual CIFAR-100 dataset index
        real_idx = self.indices[idx]
        
        img, label = self.base_dataset[real_idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dataloader(root, split_file, client_id, task_id, batch_size=32, train=True):
    # Standard augmentation for SSL (SimCLR/MAE style) could go here
    # For now, we use simple resizing to 224x224 for ViT compatibility
    transform = transforms.Compose([
        transforms.Resize(224), # ViT usually expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    dataset = FederatedCIFAR100(root, split_file, client_id, task_id, train, transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)