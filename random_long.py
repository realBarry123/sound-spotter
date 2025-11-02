from data_utils import *
import random
import numpy, torch, torchaudio, soundfile
import torch.nn.functional as F
import csv
import os

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



def create_random_long(num_occurances: int, cat: str) -> torch.Tensor:

    random_long = random.choice(list(longs))
    long_tensor = torch.load("archive_pt/" + random_long + ".pt")

    dir_path = "dataset_pt_files/" + cat
    random_short = random.choice(os.listdir(dir_path))

    print(long_tensor.shape)


    return long_tensor

    

create_random_long(5, "airplane")


    