import torch
import torch.optim as optim
from tqdm import tqdm

def train_client_ssl(
    model, 
    dataloader, 
    epochs=1, 
    lr=1e-3, 
    device='cuda'
):
    """
    Simulates local training on a client node.
    Returns: State dictionary of the trained model.
    """
    model.to(device)
    model.train()
    
    # Simple AdamW optimizer (standard for Transformers)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    epoch_loss = 0.0
    
    for epoch in range(epochs):
        batch_loss = 0.0
        count = 0
        
        # Iterating over local data
        for images, _ in dataloader: # Label is ignored in SSL
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (MAE logic inside model)
            loss, _, _ = model(images)
            
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
            count += 1
            
        epoch_loss += batch_loss / count

    # Return model weights (cpu) to free GPU memory
    return model.state_dict(), epoch_loss