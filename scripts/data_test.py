#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
from PIL import Image

BASE = Path("data/polyvore_outfits")

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_item_ids(items):
    ids = []
    for it in items:
        if isinstance(it, (str, int)):
            ids.append(str(it))
        elif isinstance(it, dict):
            for k in ("item_id", "id", "itemId"):
                if k in it:
                    ids.append(str(it[k]))
                    break
            else:
                raise ValueError(f"Could not find an id key in item: {it}")
        else:
            raise TypeError(f"Unsupported item entry type: {type(it)}")
    return ids

def extract_outfits(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "outfits" in obj and isinstance(obj["outfits"], list):
            return obj["outfits"]
        # sometimes it can be a dict keyed by set_id
        # normalize to a list of dicts with an injected set_id
        try:
            outs = []
            for k, v in obj.items():
                if isinstance(v, dict) and "items" in v:
                    vv = dict(v)
                    vv.setdefault("set_id", k)
                    outs.append(vv)
            if outs:
                return outs
        except Exception:
            pass
    raise ValueError("Unrecognized test.json structure; please open and inspect the top-level shape.")

def find_image_path(images_root: Path, item_id: str):
    exts = ("jpg", "jpeg", "png", "webp", "bmp")
    for ext in exts:
        matches = list(images_root.rglob(f"{item_id}.{ext}"))
        if matches:
            return matches[0]
    return None

def make_contact_sheet(paths, out_path: Path, thumb_size=256, padding=8, bg=(255, 255, 255)):
    if not paths:
        return
    imgs = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb_size, thumb_size))
            imgs.append(im)
        except Exception as e:
            print(f"[warn] Failed to open {p}: {e}")

    if not imgs:
        return

    # grid: roughly square
    import math
    n = len(imgs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    w = cols * thumb_size + (cols + 1) * padding
    h = rows * thumb_size + (rows + 1) * padding
    sheet = Image.new("RGB", (w, h), bg)

    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        x = padding + c * (thumb_size + padding)
        y = padding + r * (thumb_size + padding)
        sheet.paste(im, (x, y))

    sheet.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Collect all images for a single Polyvore outfit.")
    ap.add_argument("--split", default="disjoint", choices=["disjoint", "non_disjoint", "nondisjoint", "non-disjoint"],
                    help="Which split folder under data/polyvore_outfits/ to use.")
    ap.add_argument("--which", default="test", choices=["train", "valid", "test"], help="Which JSON to load.")
    ap.add_argument("--index", type=int, default=None, help="Outfit index to select.")
    ap.add_argument("--set-id", type=str, default=None, help="Outfit set_id to select (if present in JSON).")
    ap.add_argument("--copy", action="store_true", help="Copy images into outputs/outfit_<id>/")
    ap.add_argument("--sheet", action="store_true", help="Also make a contact_sheet.jpg in the output folder.")
    args = ap.parse_args()

    split_dir_map = {"non_disjoint": "non-disjoint", "nondisjoint": "non-disjoint", "non-disjoint": "non-disjoint", "disjoint": "disjoint"}
    split_dir = BASE / split_dir_map[args.split]
    json_path = split_dir / f"{args.which}.json"
    images_dir = BASE / "images"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}.")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir at {images_dir}.")

    data = load_json(json_path)
    outfits = extract_outfits(data)

    # pick outfit
    outfit = None
    chosen_label = None
    if args.set_id is not None:
        for o in outfits:
            sid = o.get("set_id") or o.get("outfit_id") or o.get("id")
            if sid is not None and str(sid) == str(args.set_id):
                outfit = o
                chosen_label = f"set_{sid}"
                break
        if outfit is None:
            raise ValueError(f"No outfit with set_id={args.set_id} found.")
    elif args.index is not None:
        if args.index < 0 or args.index >= len(outfits):
            raise IndexError(f"Index {args.index} out of range 0..{len(outfits)-1}.")
        outfit = outfits[args.index]
        sid = outfit.get("set_id") or outfit.get("outfit_id") or outfit.get("id")
        chosen_label = f"idx_{args.index}" if sid is None else f"idx_{args.index}_set_{sid}"
    else:
        # default to first
        outfit = outfits[0]
        sid = outfit.get("set_id") or outfit.get("outfit_id") or outfit.get("id")
        chosen_label = f"idx_0" if sid is None else f"idx_0_set_{sid}"

    if "items" not in outfit:
        raise KeyError("Outfit has no 'items' field; please inspect the JSON to find where item IDs live.")

    item_ids = normalize_item_ids(outfit["items"])
    print(f"Selected outfit {chosen_label} with {len(item_ids)} items.")

    # find images
    found_paths = []
    missing = []
    for iid in item_ids:
        p = find_image_path(images_dir, iid)
        if p is None:
            missing.append(iid)
        else:
            found_paths.append(p)

    print(f"Found {len(found_paths)}/{len(item_ids)} images.")
    if missing:
        print("Missing item_ids:", ", ".join(missing))

    # output
    out_dir = Path("outputs") / f"outfit_{chosen_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write a manifest text for convenience
    manifest = out_dir / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        for iid, p in zip(item_ids, [find_image_path(images_dir, iid) for iid in item_ids]):
            f.write(f"{iid}\t{p if p else 'MISSING'}\n")

    if args.copy:
        for p in found_paths:
            shutil.copy2(p, out_dir / p.name)
        print(f"Copied {len(found_paths)} images to {out_dir}")

    if args.sheet:
        sheet_path = out_dir / "contact_sheet.jpg"
        make_contact_sheet(found_paths, sheet_path)
        if sheet_path.exists():
            print(f"Wrote contact sheet to {sheet_path}")

if __name__ == "__main__":
    main()
