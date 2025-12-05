import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SimpleAudioDataset
from .model import SmallVoiceNet


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Dataset & DataLoader
    dataset = SimpleAudioDataset(args.data_root, sample_rate=args.sample_rate)
    if len(dataset) == 0:
        print("[WARN] Dataset is empty. Please add .wav files under data_root.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    n_classes = len(set(label for _, label in dataset.files)) or 2
    model = SmallVoiceNet(n_mels=64, n_classes=n_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for mel, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            mel, label = mel.to(device), label.to(device)
            # mel: (B, 1, n_mels, T)
            logits = model(mel)
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * mel.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        print(f"[INFO] Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    torch.save(model.state_dict(), args.ckpt_path)
    print(f"[INFO] Saved checkpoint to {args.ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/voice_samples")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/voice_net.pt")
    args = parser.parse_args()

    train(args)
