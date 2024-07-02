import torch

def reacall_at_k(outputs, labels, k=1):
    _, indices = torch.sort(outputs, descending=True)
    indices = indices[:, :k]
    
    return sum([labels[i] in indices[i] for i in range(indices.shape[0])])
