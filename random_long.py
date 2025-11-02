from data_utils import *
import random
import numpy, torch, torchaudio, soundfile
import torch.nn.functional as F
import csv
import os
from preprocessing_short import *

sec_len = 44100

categories = {'clapping', 'thunderstorm', 'siren', 'door_wood_knock',
               'coughing', 'airplane', 'laughing', 'drinking_sipping',
                 'helicopter', 'chainsaw', 'sneezing', 'car_horn', 'hen',
                   'toilet_flush', 'rain', 'frog', 'glass_breaking', 'vacuum_cleaner',
                     'brushing_teeth', 'crow', 'cat', 'crying_baby', 'wind',
                       'keyboard_typing', 'snoring', 'washing_machine', 'pouring_water',
                         'sheep', 'pig', 'can_opening', 'mouse_click', 'water_drops',
                           'train', 'clock_alarm', 'engine', 'hand_saw', 'breathing',
                        'cow', 'sea_waves', 'crackling_fire', 'crickets', 'fireworks',
                         'insects', 'clock_tick', 'dog', 'chirping_birds', 'footsteps',
                           'rooster', 'church_bells', 'door_wood_creaks'}

longs = {'wind', 'office', 'rain', 'construction', 'brown',
          'rainforest', 'planecabin', 'train', 'fan', 'city',
            'campfire', 'starship', 'beach', 'static', 'water',
              'engine', 'library', 'white', 'thunderstorm',
                'forest', 'river', 'factory'}


def add_tensors(T1: torch.Tensor, T2: torch.Tensor, index: int) -> torch.Tensor:
    c1, f1, t1 = T1.shape
    c2, f2, t2 = T2.shape


    if (t1 + index) <= (t2):
        T1 = F.pad(T1, (index, t2 - t1 - index))
    else:
        T1 = F.pad(T1, (index, 0))
        T2 = F.pad(T2, (0, index + t1 - t2)) 

    
    combined = T2 + T1
    return combined / combined.abs().max()

def check_interval_occupied(intervals, s, e):
    for start, end in intervals:
        if (e < end and e > start) or (s < end and s > start):
            return True
        return False

def create_random_long(num_occurances: int, cat: str) -> torch.Tensor:

    num_distractions = 4

    random_long = random.choice(list(longs))
    long_tensor = torch.load("archive_pt/" + random_long + ".pt")
    long_t_len = long_tensor.shape[2]

    dir_path = "dataset_pt_files/" + cat

    space_occupied = list()

    for i in range(num_occurances):
        random_short = random.choice(os.listdir(dir_path))
        short_tensor = torch.load(dir_path + "/" + random_short)
        short_t_len = short_tensor.shape[2]

        r = random.randint(0, long_t_len - short_t_len)
        while check_interval_occupied(space_occupied, r, r + short_t_len):
            r = random.randint(0, long_t_len - short_t_len)
        
        space_occupied.append((r, r + short_t_len))
        long_tensor = add_tensors(long_tensor, short_tensor, r)

    for i in range(num_occurances):
        dir_path = "dataset_pt_files/" + random.choice(list(categories))
        random_dist = random.choice(os.listdir(dir_path))
        dist_tensor = torch.load(dir_path + "/" + random_dist)
        dist_t_len = dist_tensor.shape[2]

        r = random.randint(0, long_t_len - dist_t_len)
        long_tensor = add_tensors(long_tensor, dist_tensor, r)

    return long_tensor

def test_create_random_long():

    random_long = create_random_long(5, "airplane")
    wave = mel_spec_to_wave(random_long)
    save_tensor_as_wav("test.wav", wave, 44100)

#test_create_random_long()




    