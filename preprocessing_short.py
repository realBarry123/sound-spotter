from data_utils import *
import random
import numpy, torch, torchaudio, soundfile
import csv
import os

global categories
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

def trim_leading_space(input_wav: torch.Tensor, threshold: float) -> torch.Tensor:
    for i in range(input_wav.shape[0]):
        if (abs(input_wav[0][i]) >= threshold):
            # Trim at dimension 1
            return input_wav[:, i:]
    return None

def trim_first(input_wav: torch.Tensor, minimum: float, maximum: float, sample_rate) -> torch.Tensor:
    out_samples = int(random.uniform(minimum, maximum) * sample_rate)
    return input_wav[:, :out_samples]

def load_wav_to_tensor(path: str) -> (torch.Tensor, int):
    wave, sample_rate = soundfile.read(path)
    wave = torch.from_numpy(wave).float()
    wave = wave.unsqueeze(0)
    return wave, int(sample_rate)

def save_tensor_as_wav(path: str, tensor: torch.Tensor, sample_rate: int):
    tensor = tensor.squeeze(0)
    soundfile.write(path, tensor, sample_rate)

def save_tensor(path: str, tensor: torch.Tensor):
    torch.save(tensor, path)

def sort_sounds():
    pass

    
def create_dict() -> dict:
    filepath = 'ESC-50-master/meta/esc50.csv'
    file_name_cat = {}

    csv_file = open(filepath, 'r', newline = '')
    with csv_file as csvfile:
        csv_reader = csv.reader(csvfile)

        header_row = True
        for row in csv_reader:
            if header_row:
                header_row = False
                continue

            key = (row[0])[:-4]
            val = row[3]

            file_name_cat[key] = val

            categories.add(val)

    print(categories)
    return file_name_cat



def preprocess():
    trim_leading_space_threshold = 0.003
    trim_first_min = 0.8
    trim_first_max = 1.0

    dict = create_dict()
    
    for cat in categories:
        dir_name = "dataset_wav_files/" + cat
        os.mkdir(dir_name)
        dir_name = "dataset_pt_files/" + cat
        os.mkdir(dir_name)


    input_path = "ESC-50-master/audio/"

    for key in dict.keys():

        tensor, sample_rate = load_wav_to_tensor(input_path + key + ".wav")
        lead_space_trimmed = trim_leading_space(tensor, trim_leading_space_threshold)

        if lead_space_trimmed is not None:
            first_trimmed = trim_first(lead_space_trimmed, trim_first_min, trim_first_max, sample_rate)


            output_wav = "dataset_wav_files/" + dict[key] + "/modified_" + key + ".wav"
            save_tensor_as_wav(output_wav, first_trimmed, sample_rate)

            output_tensor = "dataset_pt_files/" + dict[key] + "/modified_tensor_" + key + ".pt"
            save_tensor(output_tensor, tensor)




preprocess()

"""
def test_trim_leading_space():
    waveform, sample_rate = load_wav_to_tensor("ESC-50-master/audio/1-137-A-32.wav")
    print(waveform.shape)
    waveform = trim_leading_space(waveform, 0.001)
    waveform = trim_first(waveform, 1, 4, sample_rate)
    save_tensor_as_wav("test_trim_leading_space.wav", waveform, sample_rate)"""
