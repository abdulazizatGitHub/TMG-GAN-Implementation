import torch.nn.functional as F

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=1)

