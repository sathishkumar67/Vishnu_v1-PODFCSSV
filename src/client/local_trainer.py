import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    z_i, z_j: Projections of two augmented views of the same batch.
    """
    batch_size = z_i.shape[0]
    
    # Normalize inputs
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate: [2*B, Dim]
    z = torch.cat([z_i, z_j], dim=0)
    
    # Cosine similarity matrix: [2B, 2B]
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # Create mask to ignore self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, -9e15)
    
    # Positive pairs are (i, j) and (j, i)
    # We construct targets for CrossEntropy
    # Pytorch logic: sim_matrix is logits. 
    # Target for row i (view 1) is row j (view 2)
    
    labels = torch.cat([
        torch.arange(batch_size, 2*batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ], dim=0)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss