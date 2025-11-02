from data_utils import *
import random
import numpy, torch, torchaudio, soundfile
import torch.nn.functional as F
import csv
import os
from preprocessing_short import *
from metadata import longs, categories

sec_len = 44100

def add_tensors(T1: torch.Tensor, T2: torch.Tensor, index: int) -> torch.Tensor:
    c1, f1, t1 = T1.shape
    c2, f2, t2 = T2.shape
    # T1 is short, T2 is long

    if (t1 + index) <= (t2): 
        T1 = F.pad(T1, (index, t2 - t1 - index))
    else:
        T1 = F.pad(T1, (index, 0)) # pad index 0s at beginning
        T2 = F.pad(T2, (0, index + t1 - t2))  # pad index + t1 - t2 0s at end
    
    combined = T2 + T1
    return combined

def check_interval_occupied(intervals, s, e):
    for start, end in intervals:
        if (e < end and e > start) or (s < end and s > start):
            return True
        return False

def create_random_long(num_occurances: int, cat: str) -> torch.Tensor:
    num_distractions = 5
    ALPHA = 0.4 # how loud the occurances and distractions are

    random_long = random.choice(list(longs))
    long_tensor = torch.load("archive_pt/" + random_long + ".pt")
    long_t_len = long_tensor.shape[2]

    dir_path = "dataset_pt_files/" + cat

    space_occupied = list()

    # plot_spectrogram(long_tensor, "Long Tensor (before adding short)")

    for i in range(num_occurances):
        random_short = random.choice(os.listdir(dir_path))
        short_tensor = torch.load(dir_path + "/" + random_short) * ALPHA
        short_t_len = short_tensor.shape[2]

        r = random.randint(0, long_t_len - short_t_len)
        while check_interval_occupied(space_occupied, r, r + short_t_len):
            r = random.randint(0, long_t_len - short_t_len)
        
        space_occupied.append((r, r + short_t_len))
        long_tensor = add_tensors(short_tensor, long_tensor, r)

    # plot_spectrogram(long_tensor, "Long Tensor (after adding short)")

    for i in range(num_distractions):
        dir_path = "dataset_pt_files/" + random.choice(list(categories))
        random_dist = random.choice(os.listdir(dir_path))
        dist_tensor = torch.load(dir_path + "/" + random_dist) * ALPHA
        dist_t_len = dist_tensor.shape[2]

        r = random.randint(0, long_t_len - dist_t_len)
        long_tensor = add_tensors(dist_tensor, long_tensor, r)
    
    # plot_spectrogram(long_tensor, "Long Tensor (after adding distractions)")
    
    long_tensor = long_tensor / long_tensor.abs().max()

    # plot_spectrogram(long_tensor, "Long Tensor (after normalization)")

    return long_tensor

def test_create_random_long():

    random_long = create_random_long(3, "airplane")
    wave = mel_spec_to_wave(random_long)
    save_tensor_as_wav("test.wav", wave, 44100)

# test_create_random_long()
