import torch, random, os
from random_long import longs

def add_noise(short: torch.Tensor):
    noise = torch.load("archive_pt/" + random.choice(list(longs)) + ".pt")[:,:, :short.shape[2]]
    return torch.add(short, noise)

def test_add_noise():
    short = torch.zeros((1, 64, 2778))
    short = add_noise(short)
    