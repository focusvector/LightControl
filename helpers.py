import torch
def sinusoidalEmbedding(x,dim=256):
    """
    x: 1-D tensor of shape (N,) or (N,1) containing positions/timestamps.
    dim: even integer, output embedding dimension.
    returns: tensor shape (N, dim)
    """
    x = x.view(-1).float()
    if dim % 2 != 0:
        raise ValueError("sinusoidalEmbedding expects an even integer dimension.")
    half = dim//2
    freqs = torch.pow(10000.0, -torch.arange(half, device=x.device, dtype=x.dtype)/half)
    angles = x[:,None]*freqs[None,:]
    embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return embed
