import os
import torch

from .model import SmallVoiceNet


def export_onnx(
    ckpt_path: str = "checkpoints/voice_net.pt",
    onnx_path: str = "artifacts/voice_net.onnx",
):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model = SmallVoiceNet()
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint {ckpt_path} not found; exporting random-initialized model.")

    model.eval()

    # Dummy input: (B, 1, n_mels, T)
    dummy = torch.randn(1, 1, 64, 100)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch", 3: "time"}},
        opset_version=17,
    )

    print(f"[INFO] Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    export_onnx()
