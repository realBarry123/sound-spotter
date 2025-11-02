import torch, torchaudio
from model import SoundSpotter
from data_utils import *

F = 64
B = 1

MODEL_PATH = "models/model1.pt"

model = SoundSpotter(F, B).to("cpu")
state_dict, epoch = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)
model.eval()

LONG_PATH = "test_audio/clap.wav"
SHORT_PATH = "test_audio/clap_short.wav"

resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=44100)

long_wave, sample_rate = load_wav_to_tensor(LONG_PATH)
short_wave, sample_rate = load_wav_to_tensor(SHORT_PATH)
print(sample_rate)

long_wave = resampler(long_wave)[:, :, 0]
short_wave = resampler(short_wave)[:, :, 0]

long_spec = wave_to_mel_spec(long_wave).unsqueeze(0)
short_spec = wave_to_mel_spec(short_wave).unsqueeze(0)

y, heatmap = model(long_spec, short_spec)
count = round(y.item())

plot_spectrogram(long_spec.squeeze(0))
plot_heatmap(heatmap.squeeze(0))

print(count)