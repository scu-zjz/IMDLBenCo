"""Convert the official APSC-Net checkpoint to IMDLBenCo's test format.

Example:
    python -m IMDLBenCo.model_zoo.apsc_net.convert_checkpoint \
        --input ./APSC-Net.pth --output-dir ./checkpoints/apsc_net
"""

import argparse
from pathlib import Path

import torch


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint")
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        normalized[key] = value
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epoch", default=0, type=int)
    args = parser.parse_args()

    try:
        checkpoint = torch.load(
            args.input, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = normalize_state_dict(unwrap_state_dict(checkpoint))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"checkpoint-{args.epoch}.pth"
    torch.save({"model": state_dict, "epoch": args.epoch}, output)
    print(f"Saved {len(state_dict)} tensors to {output}")


if __name__ == "__main__":
    main()
