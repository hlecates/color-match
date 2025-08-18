import json, numpy as np, cv2
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PREPARE_MASKS_CONFIG

def coco_to_binary_mask(coco, img_info, ann_ids):
    h, w = img_info['height'], img_info['width']
    m = np.zeros((h, w), dtype=np.uint8)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        rle = coco.annToRLE(ann)
        dec = maskUtils.decode(rle).astype(np.uint8)
        m = np.maximum(m, dec)
    return m

def main():
    # Use configuration constants instead of command line arguments
    args = type('Args', (), PREPARE_MASKS_CONFIG)()
    
    Path(args.out_images).mkdir(parents=True, exist_ok=True)
    Path(args.out_masks).mkdir(parents=True, exist_ok=True)
    coco = COCO(args.ann)
    img_ids = coco.getImgIds()
    ok=bad=0
    for iid in img_ids:
        info = coco.loadImgs(iid)[0]
        name = info['file_name']
        src = Path(args.images)/name
        if not src.exists(): bad+=1; continue
        ann_ids = coco.getAnnIds(imgIds=[iid])
        mask = coco_to_binary_mask(coco, info, ann_ids)
        # write copies to seg_data
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None: bad+=1; continue
        cv2.imwrite(str(Path(args.out_images)/name), img)
        cv2.imwrite(str(Path(args.out_masks)/Path(name).with_suffix(".png")), (mask*255).astype(np.uint8))
        ok+=1
    print(f"[modanet] wrote {ok} pairs, skipped {bad}")

if __name__ == "__main__":
    main()
