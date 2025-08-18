#!/usr/bin/env python3
import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SEGMENT_TRAIN_CONFIG

class SegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=384, split_ratio=0.9, train=True, seed=42):
        self.images = sorted([p for p in Path(images_dir).glob("*.*")])
        self.masks  = {p.stem: p for p in Path(masks_dir).glob("*.png")}
        random.Random(seed).shuffle(self.images)
        cut = int(len(self.images)*split_ratio)
        self.images = self.images[:cut] if train else self.images[cut:]
        self.size = size
        self.train = train

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        ip = self.images[i]
        mp = self.masks.get(ip.stem)
        img = Image.open(ip).convert("RGB")
        if mp is None:
            m = Image.new("L", img.size, 0)
        else:
            m = Image.open(mp).convert("L")
        # Resize shorter side to size, keep aspect, then center-crop square
        img = tv.transforms.functional.resize(img, self.size, interpolation=tv.transforms.InterpolationMode.BILINEAR)
        m   = tv.transforms.functional.resize(m,   self.size, interpolation=tv.transforms.InterpolationMode.NEAREST)
        img = tv.transforms.functional.center_crop(img, [self.size, self.size])
        m   = tv.transforms.functional.center_crop(m,   [self.size, self.size])
        if self.train:
            if random.random() < 0.5:
                img = tv.transforms.functional.hflip(img); m = tv.transforms.functional.hflip(m)
            # light color jitter helps robustness of masks
            img = tv.transforms.ColorJitter(0.2,0.2,0.1,0.05)(img)
        x = tv.transforms.ToTensor()(img)           # [3,H,W] 0..1
        y = torch.from_numpy(np.array(m) > 127).float().unsqueeze(0)  # [1,H,W] {0,1}
        return x, y

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2*(probs*targets).sum(dim=(2,3))
    den = (probs+targets).sum(dim=(2,3))+eps
    return 1 - (num/den).mean()

def main():
    # Use configuration constants instead of command line arguments
    args = type('Args', (), SEGMENT_TRAIN_CONFIG)()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # model
    model = tv.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 1, 1)  # binary logits
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_ds = SegDataset(args.images, args.masks, size=args.size, train=True)
    val_ds   = SegDataset(args.images, args.masks, size=args.size, train=False)
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_miou = 0.0
    for ep in range(1, args.epochs+1):
        model.train(); tl=0.0
        for x,y in tr:
            x,y = x.to(device), y.to(device)
            out = model(x)["out"]
            bce = F.binary_cross_entropy_with_logits(out, y)
            dce = dice_loss(out, y)
            loss = bce + 0.5*dce
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*x.size(0)
        # val IoU
        model.eval(); inter=0; union=0
        with torch.no_grad():
            for x,y in va:
                x,y = x.to(device), y.to(device)
                p = (torch.sigmoid(model(x)["out"])>0.5).float()
                inter += (p*y).sum().item()
                union += ((p+y)>0).float().sum().item()
        miou = inter/union if union>0 else 0.0
        print(f"epoch {ep:03d}  train_loss={tl/len(train_ds):.4f}  mIoU={miou:.3f}")
        torch.save(model.state_dict(), f"{args.out}/deeplabv3_mnv3_ep{ep:03d}.pt")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), f"{args.out}/best.pt")
    print("best mIoU:", best_miou)

if __name__ == "__main__":
    main()
