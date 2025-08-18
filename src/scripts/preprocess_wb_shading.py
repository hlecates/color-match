#!/usr/bin/env python3
import cv2, numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PREPROCESS_WB_SHADING_CONFIG

def srgb_to_lin(u8):
    x = u8.astype(np.float32)/255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)
def lin_to_srgb(lin):
    a = 0.055
    x = np.clip(lin, 0, 1)
    y = np.where(x <= 0.0031308, x*12.92, (1+a)*np.power(x, 1/2.4) - a)
    return np.clip(y*255.0, 0, 255).astype(np.uint8)

def gray_world_on_mask(img_bgr, mask):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lin = srgb_to_lin(img_rgb) # HxWx3
    m = mask.astype(bool)
    if m.sum() < 10:  # too small
        return img_bgr
    means = lin[m].reshape(-1,3).mean(axis=0)
    target = means.mean()
    scale = np.clip(target / (means + 1e-6), 0.5, 2.0)  # clamp extreme
    corr = lin * scale
    out = cv2.cvtColor(lin_to_srgb(corr), cv2.COLOR_RGB2BGR)
    return out

def shading_norm(img_bgr, mask, sigma=31, eps=1e-6):
    """
    Estimate smooth illumination on L channel inside mask and divide.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(img_lab)
    m = (mask>0).astype(np.uint8)

    # Smooth illumination field (Gaussian on masked L)
    L_blur = cv2.GaussianBlur(L, (0,0), sigmaX=sigma, sigmaY=sigma)
    # avoid zeros; use mask mean for background to prevent division
    meanL = max(1.0, L[m>0].mean() if m.sum()>0 else 128.0)
    illum = L_blur
    illum[illum<5] = meanL
    # Normalize
    L_corr = L * (meanL/(illum+eps))
    # Clamp
    L_corr = np.clip(L_corr, 0, 255)

    out_lab = cv2.merge([L_corr, A, B]).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    return out_bgr

def bbox_from_mask(mask):
    ys, xs = np.where(mask>0)
    if len(xs)==0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def crop_pad_resize(img, mask, size=256, margin=0.06):
    h,w = mask.shape
    bb = bbox_from_mask(mask)
    if bb is None:
        # fallback: center square
        side=min(h,w); y0=(h-side)//2; x0=(w-side)//2
        img_c = img[y0:y0+side, x0:x0+side]; m_c=mask[y0:y0+side, x0:x0+side]
    else:
        x0,y0,x1,y1 = bb
        dx=int((x1-x0+1)*margin); dy=int((y1-y0+1)*margin)
        x0=max(0,x0-dx); y0=max(0,y0-dy); x1=min(w-1,x1+dx); y1=min(h-1,y1+dy)
        img_c = img[y0:y1+1, x0:x1+1]; m_c = mask[y0:y1+1, x0:x1+1]
    hh,ww = img_c.shape[:2]; side=max(hh,ww)
    pad_y = side-hh; pad_x = side-ww
    top=pad_y//2; bottom=pad_y-top; left=pad_x//2; right=pad_x-left
    img_sq = cv2.copyMakeBorder(img_c, top,bottom,left,right, cv2.BORDER_CONSTANT, value=(0,0,0))
    m_sq   = cv2.copyMakeBorder(m_c,   top,bottom,left,right, cv2.BORDER_CONSTANT, value=0)
    img_r  = cv2.resize(img_sq, (size,size), interpolation=cv2.INTER_AREA)
    m_r    = cv2.resize(m_sq, (size,size), interpolation=cv2.INTER_NEAREST)
    return img_r, m_r

def main():
    # Use configuration constants instead of command line arguments
    args = type('Args', (), PREPROCESS_WB_SHADING_CONFIG)()
    
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ok=bad=0
    for line in Path(args.items_list).read_text().splitlines():
        ip = Path(line.strip())
        mp = Path(args.masks_dir)/(ip.stem+".png")
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        m   = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            print("[skip] missing", ip); bad+=1; continue

        # WB on masked pixels
        if args.wb == "grayworld":
            img = gray_world_on_mask(img, m)
        # Shading normalization
        if args.shading == "on":
            img = shading_norm(img, m, sigma=31)

        # Crop/pad/resize & apply mask to black background
        img_r, m_r = crop_pad_resize(img, m, size=args.size)
        img_r[m_r==0] = (0,0,0)

        cv2.imwrite(str(out_dir/(ip.stem+".png")), img_r)
        ok+=1
    print(f"[prep] wrote {ok}, skipped {bad}")

if __name__ == "__main__":
    main()
