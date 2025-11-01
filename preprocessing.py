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
    out_samples = random.randrange(minimum, maximum) * sample_rate
    return input_wav[:, :out_samples]

def load_wav_to_tensor(path: str) -> (torch.Tensor, int):
    wave, sample_rate = soundfile.read(path)
    wave = torch.from_numpy(wave).float()
    wave = wave.unsqueeze(0)
    return wave, sample_rate

def save_tensor_as_wav(path: str, tensor: torch.Tensor, sample_rate: int):
    tensor = tensor.squeeze(0)
    soundfile.write(path, tensor, sample_rate)


def sort_sounds():
    #pass

    file_name_cat = {}
    categories = set()

    filepath = 'ESC-50-master\ESC-50-master\meta\esc50.csv'
    csv_file = open(filepath, 'r', newline = '')
    with csv_file as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            key = row[0]
            val = row[3]

            file_name_cat[key] = val

            categories.add(val)

    print(categories)
    
    """{'clapping', 'breathing', 'category', 'insects', 'thunderstorm', 'church_bells',
      'hen', 'chirping_birds', 'fireworks', 'clock_alarm', 'sheep', 'water_drops',
        'snoring', 'car_horn', 'chainsaw', 'toilet_flush', 'pig', 'clock_tick',
        'crying_baby', 'sea_waves', 'airplane', 'siren', 'sneezing', 'coughing',
        'washing_machine', 'dog', 'can_opening', 'wind', 'engine', 'frog', 'door_wood_creaks',
        'rooster', 'footsteps', 'crackling_fire', 'pouring_water', 'cat', 'drinking_sipping',
        'helicopter', 'door_wood_knock', 'train', 'rain', 'glass_breaking', 'mouse_click',
        'hand_saw', 'cow', 'crickets', 'vacuum_cleaner', 'laughing', 'brushing_teeth', 'crow',
        'keyboard_typing'}
    """



sort_sounds()
    

def test_trim_leading_space():
    waveform, sample_rate = load_wav_to_tensor("ESC-50-master/audio/1-137-A-32.wav")
    print(waveform.shape)
    waveform = trim_leading_space(waveform, 0.001)
    waveform = trim_first(waveform, 1, 4, sample_rate)
    save_tensor_as_wav("test_trim_leading_space.wav", waveform, sample_rate)
