from typing import List, Tuple
import os
import glob
import torch
import torchaudio


class SimpleAudioDataset(torch.utils.data.Dataset):
    """
    Very simple dataset for voice fine-tuning.

    Expected directory layout:

        data_root/
            class0/*.wav
            class1/*.wav
            ...

    Each subfolder name is treated as a label.
    """

    def __init__(self, data_root: str, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.files: List[Tuple[str, int]] = []
        self.label_to_idx = {}

        # Discover classes & files
        class_names = sorted(
            [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        )

        for idx, cname in enumerate(class_names):
            self.label_to_idx[cname] = idx
            class_dir = os.path.join(data_root, cname)
            for wav in glob.glob(os.path.join(class_dir, "*.wav")):
                self.files.append((wav, idx))

        if not self.files:
            print(f"[WARN] No .wav files found in {data_root}. Dataset will be empty.")

        # Define mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=64,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path, label = self.files[index]
        waveform, sr = torchaudio.load(path)

        # Resample to target SR if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # (1, T)
        mel = self.mel_transform(waveform)  # (1, n_mels, time)

        # normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, label
