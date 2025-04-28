import torch
import torch.nn.functional as F

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=1)

def pairwise_cosine_similarity(A, B):
    """
    Returns a matrix of cosine similarities between all rows of A and all rows of B.
    """
    A_norm = F.normalize(A, dim=1)
    B_norm = F.normalize(B, dim=1)
    return torch.matmul(A_norm, B_norm.T)  # shape: [batch_size, batch_size]