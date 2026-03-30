import argparse
from pathlib import Path

import torch

from network_local import NetworkT40Local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = NetworkT40Local().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 32, 32, device=device)

    out_path = Path(args.out) if args.out else Path(args.model).with_suffix(".onnx")

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )

    print(f"Exported ONNX to {out_path}")


if __name__ == "__main__":
    main()