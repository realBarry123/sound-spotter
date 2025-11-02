from data_utils import *
import random
import numpy, torch, torchaudio, soundfile
import csv

def trim_leading_space(input_wav: torch.Tensor, threshold: float) -> torch.Tensor:
    for i in range(input_wav.shape[0]):
        if (abs(input_wav[0][i]) >= threshold):
            # Trim at dimension 1
            return input_wav[:, i:]
    return input_wav

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


def sort_sounds():
    pass

    
def create_dict() -> dict:
    filepath = 'ESC-50-master/ESC-50-master/meta/esc50.csv'
    file_name_cat = {}

    csv_file = open(filepath, 'r', newline = '')
    with csv_file as csvfile:
        csv_reader = csv.reader(csvfile)

        header_row = True
        for row in csv_reader:
            if header_row:
                header_row = False
                continue

            key = row[0]
            val = row[3]

            file_name_cat[key] = val

    return file_name_cat



def preprocess():
    trim_leading_space_threshold = 0.001
    trim_first_min = 0.8
    trim_first_max = 1.0

    dict = create_dict()
    
    input_path = "ESC-50-master/ESC-50-master/audio/"
    output_path = "dataset_processed/"

    for key in dict.keys():

        tensor, sample_rate = load_wav_to_tensor(input_path + key)
        lead_space_trimmed = trim_leading_space(tensor, trim_leading_space_threshold)
        first_trimmed = trim_first(lead_space_trimmed, trim_first_min, trim_first_max, sample_rate)
        
        output = output_path + dict[key] + "/modified_" + key
        save_tensor_as_wav(output, first_trimmed, sample_rate)


preprocess()

"""
def test_trim_leading_space():
    waveform, sample_rate = load_wav_to_tensor("ESC-50-master/audio/1-137-A-32.wav")
    print(waveform.shape)
    waveform = trim_leading_space(waveform, 0.001)
    waveform = trim_first(waveform, 1, 4, sample_rate)
    save_tensor_as_wav("test_trim_leading_space.wav", waveform, sample_rate)"""
