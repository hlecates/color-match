#!/usr/bin/env python3
import onnxruntime as ort, numpy as np, cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SEGMENT_PAIRS_CONFIG

def preprocess(img, size):
    h,w = img.shape[:2]
    s = size
    # Resize short side to s, keep aspect, then center-crop square s×s
    scale = s / min(h,w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (nh - s)//2; x0 = (nw - s)//2
    img_c = img_r[y0:y0+s, x0:x0+s]
    x = (img_c[..., ::-1].astype(np.float32)/255.0)  # BGR->RGB, 0..1
    x = np.transpose(x, (2,0,1))[None, ...]          # [1,3,s,s]
    return x

def postprocess(logits, orig_shape):
    prob = 1/(1+np.exp(-logits))[0,0]  # [s,s]
    m = (prob > 0.5).astype(np.uint8)
    # optional morphology
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    # upscale back to original
    m = cv2.resize(m, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    # largest component
    n,labels = cv2.connectedComponents(m)
    if n>1:
        counts = np.bincount(labels.ravel()); counts[0]=0
        m = (labels == counts.argmax()).astype(np.uint8)
    return m

def main():
    # Use configuration constants instead of command line arguments
    args = type('Args', (), SEGMENT_PAIRS_CONFIG)()
    
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ok=bad=0
    for line in Path(args.items_list).read_text().splitlines():
        p = Path(line.strip())
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None: bad+=1; continue
        x = preprocess(img, args.size)
        logits = sess.run(None, {"image": x})[0]
        m = postprocess(logits, img.shape[:2])
        cv2.imwrite(str(out_dir/(p.stem+".png")), (m*255).astype(np.uint8))
        ok+=1
    print(f"[seg] wrote {ok} masks, skipped {bad}")

if __name__ == "__main__":
    main()
