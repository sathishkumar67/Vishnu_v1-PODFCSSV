import torch
import copy

def fed_avg(global_model, client_weights_list):
    """
    Aggregates client weights using Federated Averaging.
    
    global_model: The base model architecture
    client_weights_list: List of state_dicts from clients
    """
    # Create a deep copy to avoid modifying the original during calculation
    new_weights = copy.deepcopy(global_model.state_dict())
    
    num_clients = len(client_weights_list)
    
    # Iterate over every parameter in the model (e.g., layer1.weight, layer2.bias)
    for key in new_weights.keys():
        # Initialize tensor to zero
        avg_tensor = torch.zeros_like(new_weights[key], dtype=torch.float32)
        
        # Sum up weights from all clients
        for client_w in client_weights_list:
            avg_tensor += client_w[key]
            
        # Divide by number of clients
        new_weights[key] = avg_tensor / num_clients
        
    return new_weights