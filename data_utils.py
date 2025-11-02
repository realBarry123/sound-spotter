import torch, torchaudio, os, tqdm, gc
import soundfile
import matplotlib.pyplot as plt

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

def plot_heatmap(heatmap: torch.Tensor, title: str="Heatmap"):
    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap.detach().numpy(), cmap='magma', aspect='auto')
    plt.title(title)
    plt.yticks([])  # remove y-axis
    plt.xlabel("Time / Index")
    plt.colorbar(label="Intensity")
    plt.show()


def wave_to_mel_spec(waveform: torch.Tensor, n_fft=1024, hop_length=256, n_mels=64):
    # Note: upsamples by approx hop_length

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
    # Note: upsamples by approx hop_length
    
    # Undo log scaling
    mel_spec = torch.expm1(mel_spec)  # log1p -> x
    
    # Convert Mel to linear STFT
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=16000
    )
    lin_spec = inv_mel(mel_spec)  # [C, F, T]
    
    # Griffin-Lim to estimate phase
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, window_fn=torch.hann_window, n_iter=n_iter, power=1
    )
    waveform = griffin_lim(lin_spec)  # [C, T]
    
    return waveform

def test_mel_reconstruction():
    wav, sample_rate = load_wav_to_tensor("test_in.wav")
    spec = wave_to_mel_spec(wav)
    wav = mel_spec_to_wave(spec)
    print(sample_rate)
    save_tensor_as_wav("test_out.wav", wav, sample_rate)

# test_mel_reconstruction()

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
