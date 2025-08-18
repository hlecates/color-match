#!/usr/bin/env python3
import torch, torch.nn as nn, torchvision as tv
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import EXPORT_TO_ONNX_CONFIG

def build_model():
    m = tv.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    m.classifier[-1] = nn.Conv2d(m.classifier[-1].in_channels, 1, 1)
    return m

def main():
    # Use configuration constants instead of command line arguments
    args = type('Args', (), EXPORT_TO_ONNX_CONFIG)()
    
    m = build_model()
    sd = torch.load(args.ckpt, map_location="cpu")
    m.load_state_dict(sd); m.eval()

    dummy = torch.randn(1,3,args.size,args.size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image":{0:"B"}, "logits":{0:"B"}})
    print("wrote", args.out)

if __name__ == "__main__":
    main()
