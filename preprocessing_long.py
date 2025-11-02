import random
import numpy, torch, torchaudio, soundfile
import csv
import os
from data_utils import *

global longs
longs = {'wind', 'office', 'rain', 'construction', 'brown',
          'rainforest', 'planecabin', 'train', 'fan', 'city',
            'campfire', 'starship', 'beach', 'static', 'water',
              'engine', 'library', 'white', 'thunderstorm',
                'forest', 'river', 'factory'}

def save_wavs_as_tensors():

    dir = "archive/"
    output_dir = "archive_pt/"

    for filename in os.listdir(dir):
        wave, sample_rate = soundfile.read(dir + filename)
        wave = torch.from_numpy(wave).float()
        wave = wave.unsqueeze(0)
        tensor = wave_to_mel_spec(wave)

        torch.save(tensor, output_dir + filename[:-4] + ".pt")

save_wavs_as_tensors()
    
