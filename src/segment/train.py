#!/usr/bin/env python3
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import sys
import time
from tqdm.auto import tqdm
sys.path.append(str(Path(__file__).parent.parent))
from config import SEGMENT_TRAIN_CONFIG

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2*(probs*targets).sum(dim=(2,3))
    den = (probs+targets).sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

class SegDatasetPattern(Dataset):
    def __init__(self, images_dir, masks_dir, img_prefix="img", mask_prefix="seg",
                 ext=(".jpg",".jpeg",".png"), size=384, train=True, split_ratio=0.9, seed=42):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.exts = tuple(e.lower() for e in ext) if isinstance(ext, (list, tuple)) else (ext.lower(),)
        all_imgs = sorted([p for p in self.images_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in self.exts and p.stem.startswith(self.img_prefix)])
        # keep only those that have a corresponding mask
        self.pairs = []
        for ip in all_imgs:
            num = ip.stem[len(self.img_prefix):]  # "0001" from "img0001"
            mp = self.masks_dir / f"{self.mask_prefix}{num}{ip.suffix}"
            if not mp.exists():
                # try alternate ext if needed
                alt = None
                for e in self.exts:
                    cand = self.masks_dir / f"{self.mask_prefix}{num}{e}"
                    if cand.exists(): alt = cand; break
                mp = alt if alt is not None else None
            if mp is not None:
                self.pairs.append((ip, mp))
        # split
        rng = random.Random(seed)
        rng.shuffle(self.pairs)
        cut = int(len(self.pairs) * split_ratio)
        self.pairs = self.pairs[:cut] if train else self.pairs[cut:]
        self.size = size
        self.train = train
        # transforms
        self.color_jitter = tv.transforms.ColorJitter(0.2, 0.2, 0.1, 0.05)

    def __len__(self): return len(self.pairs)

    def _resize_center_square(self, img, size, interp):
        img = tv.transforms.functional.resize(img, size, interpolation=interp)
        return tv.transforms.functional.center_crop(img, [size, size])

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        m   = Image.open(mp).convert("L")  # JPEG masks may have compression shades
        # resize & crop
        img = self._resize_center_square(img, self.size, tv.transforms.InterpolationMode.BILINEAR)
        m   = self._resize_center_square(m,   self.size, tv.transforms.InterpolationMode.NEAREST)
        # augment
        if self.train:
            if random.random() < 0.5:
                img = tv.transforms.functional.hflip(img)
                m   = tv.transforms.functional.hflip(m)
            img = self.color_jitter(img)
        # to tensor
        x = tv.transforms.ToTensor()(img)                      # [3,H,W] in 0..1
        y = (torch.from_numpy(np.array(m)) > 127).float().unsqueeze(0)  # [1,H,W] {0,1}
        return x, y

def build_model():
    m = tv.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    # change classifier to 1 channel (binary)
    m.classifier[-1] = nn.Conv2d(m.classifier[-1].in_channels, 1, 1)
    return m

def main():
    # Use configuration constants instead of command line arguments
    args = type('Args', (), SEGMENT_TRAIN_CONFIG)()
    
    exts = tuple(s.strip() for s in args.ext.split(",") if s.strip())
    Path(args.out).mkdir(parents=True, exist_ok=True)

    train_ds = SegDatasetPattern(args.images, args.masks,
                                 img_prefix=args.img_prefix, mask_prefix=args.mask_prefix,
                                 ext=exts, size=args.size, train=True,  split_ratio=args.split_ratio, seed=args.seed)
    val_ds   = SegDatasetPattern(args.images, args.masks,
                                 img_prefix=args.img_prefix, mask_prefix=args.mask_prefix,
                                 ext=exts, size=args.size, train=False, split_ratio=args.split_ratio, seed=args.seed)

    print(f"train samples: {len(train_ds)} | val samples: {len(val_ds)}")

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    va = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_miou = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, running_bce, running_dice, seen = 0.0, 0.0, 0.0, 0

        pbar = tqdm(tr, desc=f"Epoch {ep}/{args.epochs}", leave=False)
        for step, (x, y) in enumerate(pbar, 1):
            x, y = x.to(device), y.to(device)

            out = model(x)["out"]
            bce = F.binary_cross_entropy_with_logits(out, y)
            dce = dice_loss(out, y)
            loss = bce + 0.5 * dce

            opt.zero_grad()
            loss.backward()
            opt.step()

            # running means
            bs = x.size(0)
            seen += bs
            running      += loss.item() * bs
            running_bce  += bce.item()  * bs
            running_dice += dce.item()  * bs

            # lightweight batch display
            if step % 50 == 0 or step == 1:
                pbar.set_postfix(loss=running/seen, bce=running_bce/seen, dice=running_dice/seen)

        train_loss = running / max(1, seen)
        train_bce  = running_bce / max(1, seen)
        train_dice = running_dice / max(1, seen)

        model.eval()
        inter = union = 0.0
        tp = fp = fn = 0.0
        val_dice_accum = 0.0
        val_seen = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                logits = model(x)["out"]
                probs  = torch.sigmoid(logits)
                pred   = (probs > 0.5).float()

                inter += (pred * y).sum().item()
                union += ((pred + y) > 0).float().sum().item()

                tp += (pred * y).sum().item()
                fp += (pred * (1 - y)).sum().item()
                fn += ((1 - pred) * y).sum().item()

                # dice per-batch
                val_dice_accum += (2.0 * (pred * y).sum().item()) / (pred.sum().item() + y.sum().item() + 1e-6)
                val_seen += 1

        miou  = inter / union if union > 0 else 0.0
        v_dice = val_dice_accum / max(1, val_seen)
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        epoch_s = time.time() - t0

        improved = ""
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), f"{args.out}/best.pt")
            improved = "  ↑ best"

        print(
            f"ep {ep:03d} | train loss {train_loss:.4f} "
            f"(bce {train_bce:.4f}, dice {train_dice:.4f}) | "
            f"val mIoU {miou:.3f}  Dice {v_dice:.3f}  P {prec:.3f}  R {rec:.3f} | "
            f"{epoch_s:.1f}s{improved}"
        )
        torch.save(model.state_dict(), f"{args.out}/deeplabv3_mnv3_ep{ep:03d}.pt")


if __name__ == "__main__":
    main()
