
import torch, torchaudio, os, tqdm, shutil, random, gc
#import matplotlib.pyplot as plt

import torch
import torchaudio

def wave_to_mel_spec(waveform: torch.Tensor, n_fft=1024, hop_length=256, n_mels=64):
    
    #Convert waveform [C, T] into log-Mel spectrogram [C, n_mels, T]
    # Compute STFT

    spec = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length,
        return_complex=True, window=torch.hann_window(n_fft)
    )  # [C, F=n_fft/2+1, T]
    
    # Convert to magnitude
    mag = spec.abs()  # [C, F, T]
    
    # Apply Mel filterbank
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=n_mels, sample_rate=16000, n_stft=mag.size(1)
    )
    mel_spec = mel_scale(mag)  # [C, n_mels, T]
    
    # Log scaling (optional but common)
    log_mel_spec = torch.log1p(mel_spec)
    
    return log_mel_spec  # [C, n_mels, T]

def mel_spec_to_wave(mel_spec, n_fft=1024, hop_length=256, n_mels=64, n_iter=32):
    
    #Convert log-Mel spectrogram [C, n_mels, T] back to waveform [C, T]
    #(approximate using Griffin-Lim)
    
    # Undo log scaling
    mel_spec = torch.expm1(mel_spec)  # log1p -> x
    
    # Convert Mel to linear STFT
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=16000
    )
    lin_spec = inv_mel(mel_spec)  # [C, F, T]
    
    # Griffin-Lim to estimate phase
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, window_fn=torch.hann_window, n_iter=n_iter
    )
    waveform = griffin_lim(lin_spec)  # [C, T]
    
    return waveform
'''
def load_wav_to_mel_spec(path: str) -> torch.Tensor: 
    Load a waveform from the directory `path` and return a spectrogram (size [C, F, T]).
    waveform, sample_rate = torchaudio.load(path)
    spec = wave_to_mel_spec(waveform)
    return spec

def save_mel_spec_as_wav(spec: torch.Tensor, path: str, sample_rate: int=44100):
    Save a spectrogram as a `.wav` file in directory `path`.
    waveform = mel_spec_to_wave(spec)
    if len(list(waveform.shape)) == 1:
        waveform = torch.unsqueeze(waveform, 0)
    torchaudio.save(
        uri=path,
        src=waveform,
        sample_rate=sample_rate
    )
'''
'''
def plot_spectrogram(spec: torch.Tensor, title: str="Spectrogram") -> None:
    if isinstance(spec, torch.Tensor):
        spec = spec.squeeze(0).cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency (Mel bins)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()
'''
def wavs_to_tensors(path_in: str, path_out: str) -> None:
    #Convert all `.wav` files in directory `path_in` to `.pt` files in directory `path_out`.
    wav_files = [f for f in os.listdir(path_in) if f.endswith('.wav')]
    for filename in tqdm.tqdm(wav_files):
        if filename.endswith('.wav'):
            input_path = os.path.join(path_in, filename)
            output_filename = os.path.splitext(filename)[0] + '.pt'
            output_path = os.path.join(path_out, output_filename)

            #tensor = load_wav(input_path)
            torch.save(tensor, output_path)

            # Explicit memory cleanup
            del tensor
            gc.collect()
