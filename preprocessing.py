from data_utils import *
import random
import numpy, torch, torchaudio, soundfile

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
    pass

def test_trim_leading_space():
    waveform, sample_rate = load_wav_to_tensor("ESC-50-master/audio/1-137-A-32.wav")
    print(waveform.shape)
    waveform = trim_leading_space(waveform, 0.001)
    waveform = trim_first(waveform, 1, 4, sample_rate)
    save_tensor_as_wav("test_trim_leading_space.wav", waveform, sample_rate)
