#!/usr/bin/env python3
import json, random, itertools, csv, sys
from pathlib import Path
from collections import defaultdict, Counter
sys.path.append(str(Path(__file__).parent.parent))
from config import MAKE_PAIRS_CONFIG

# Optional: color-similar hard negatives (requires Pillow)
try:
    from PIL import Image
    import colorsys
    HAS_PIL = True
except Exception:
    HAS_PIL = False

RNG = random.Random()

def canonical_pair_key(a_id, b_id):
    return (a_id, b_id) if a_id <= b_id else (b_id, a_id)

def parse_allowed_pairs(s: str | None):
    if not s: return None
    out = set()
    for tok in s.split(","):
        a,b = tok.strip().split(":")
        out.add(frozenset([a.lower(), b.lower()]))
    return out

def load_split_sets(disjoint_dir: Path, split: str):
    p = disjoint_dir / f"{split}.json"
    data = json.loads(p.read_text())
    # Expected each entry: {"set_id": "...", "items": [{"item_id":"...", "index": ...}, ...]}
    return [{"set_id": str(d["set_id"]),
             "items": [str(it["item_id"]) for it in d["items"]]} for d in data]

def load_categories(categories_csv: Path):
    # categories.csv: usually "index,name" (id -> text)
    cat_map = {}
    with categories_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row: continue
            try:
                idx = int(row[0])
                name = row[1].strip().lower()
                cat_map[idx] = name
            except Exception:
                continue
    return cat_map

def load_item_metadata(meta_path: Path):
    """
    polyvore_item_metadata.json can be a dict[item_id]->obj or a list of objects.
    We return: item_id -> dict with possible keys {category_id|categoryid|category_name|semantic_category}
    """
    meta_raw = json.loads(meta_path.read_text())
    meta = {}
    if isinstance(meta_raw, dict):
        for k,v in meta_raw.items():
            meta[str(k)] = v
    else:
        for v in meta_raw:
            if "item_id" in v:
                meta[str(v["item_id"])] = v
    return meta

def item_category(item_id: str, meta: dict, cat_map: dict):
    m = meta.get(item_id, {}) if meta else {}
    # Try id-based first
    cid = m.get("category_id", m.get("categoryid", m.get("categoryId")))
    if isinstance(cid, str) and cid.isdigit():
        cid = int(cid)
    if isinstance(cid, int) and cid in cat_map:
        return cat_map[cid]
    # Fallbacks
    for k in ("category_name", "semantic_category", "category"):
        if k in m and isinstance(m[k], str) and m[k].strip():
            return m[k].strip().lower()
    return "unknown"

def resolve_image_path(images_dir: Path, item_id: str):
    # Try common extensions; don’t fail if missing—just use .jpg
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{item_id}{ext}"
        if p.exists(): return str(p)
    return str(images_dir / f"{item_id}.jpg")

def mean_hsv_quick(img_path: str, thumb=64):
    img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
    px = img.getdata()
    H=S=V=0.0
    n=0
    for r,g,b in px:
        h,s,v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        H+=h; S+=s; V+=v; n+=1
    return (H/n, S/n, V/n)

def hsv_distance(h1, h2):
    dh = min(abs(h1[0]-h2[0]), 1.0-abs(h1[0]-h2[0]))
    ds = abs(h1[1]-h2[1])
    dv = abs(h1[2]-h2[2])
    return (dh*2.0) + ds + (dv*0.5)

def build_indices(sets, images_dir: Path, meta: dict, cat_map: dict):
    items_by_set = {}
    items_by_cat = defaultdict(list)  # cat -> list of (item_id, set_id, img_path)
    item_img = {}
    item_cat = {}

    for s in sets:
        sid = s["set_id"]
        ids = s["items"]
        items_by_set[sid] = ids
        for iid in ids:
            if iid not in item_img:
                item_img[iid] = resolve_image_path(images_dir, iid)
                item_cat[iid] = item_category(iid, meta, cat_map)
            items_by_cat[item_cat[iid]].append((iid, sid, item_img[iid]))

    return items_by_set, items_by_cat, item_img, item_cat

def generate_positive_pairs(sets, item_cat, item_img, allowed_pairs, seen_keys):
    rows = []
    for s in sets:
        sid = s["set_id"]
        ids = s["items"]
        for a, b in itertools.combinations(ids, 2):
            a_cat = item_cat[a]; b_cat = item_cat[b]
            if allowed_pairs and frozenset([a_cat, b_cat]) not in allowed_pairs:
                continue
            key = canonical_pair_key(a, b)
            if key in seen_keys:  # avoid duplicates/mirrors
                continue
            seen_keys.add(key)
            rows.append({
                "a_id": a, "b_id": b,
                "a_img": item_img[a], "b_img": item_img[b],
                "a_cat": a_cat, "b_cat": b_cat,
                "set_a": sid, "set_b": sid,
                "label": 1
            })
    return rows

def generate_negative_pairs(pos_rows, items_by_cat, neg_per_pos, hard_negatives):
    neg_rows, seen = [], set()
    color_cache = {} if (hard_negatives and HAS_PIL) else None

    def pick_candidates(anchor_img, target_cat, forbid_set_ids, forbid_item_ids):
        cand = [(iid, sid, img) for (iid, sid, img) in items_by_cat[target_cat]
                if sid not in forbid_set_ids and iid not in forbid_item_ids]
        if not cand: return []
        RNG.shuffle(cand)
        return cand

    for r in pos_rows:
        need = max(1, int(round(neg_per_pos)))
        # Strategy: fix A, sample B' of same cat as B from other sets
        forbid_sets = {r["set_a"]}
        forbid_items = {r["a_id"], r["b_id"]}
        cand = pick_candidates(r["a_img"], r["b_cat"], forbid_sets, forbid_items)

        if hard_negatives and cand and HAS_PIL:
            try:
                anc_sig = color_cache.get(r["a_img"])
                if anc_sig is None:
                    anc_sig = mean_hsv_quick(r["a_img"])
                    color_cache[r["a_img"]] = anc_sig
                scored = []
                for iid, sid, img in cand:
                    s = color_cache.get(img)
                    if s is None:
                        s = mean_hsv_quick(img); color_cache[img] = s
                    scored.append((hsv_distance(anc_sig, s), iid, sid, img))
                scored.sort(key=lambda x: x[0])  # closer = harder
                cand = [(iid, sid, img) for _, iid, sid, img in scored]
            except Exception:
                pass

        for iid, sid, img in cand[:need]:
            key = canonical_pair_key(r["a_id"], iid)
            if key in seen: continue
            seen.add(key)
            neg_rows.append({
                "a_id": r["a_id"], "b_id": iid,
                "a_img": r["a_img"], "b_img": img,
                "a_cat": r["a_cat"], "b_cat": r["b_cat"],
                "set_a": r["set_a"], "set_b": sid,
                "label": 0,
                "hard": bool(hard_negatives)
            })

    return neg_rows

def write_jsonl(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, start=1):
            out = {
                "id": f"p_{i:06d}",
                "a_img": r["a_img"], "b_img": r["b_img"],
                "label": int(r["label"]),
                "a_cat": r["a_cat"], "b_cat": r["b_cat"],
                "a_id": r["a_id"], "b_id": r["b_id"],
                "set_a": r["set_a"], "set_b": r["set_b"]
            }
            if "hard" in r: out["hard"] = bool(r["hard"])
            f.write(json.dumps(out) + "\n")

def main():
    # Use configuration constants instead of command line arguments
    args = type('Args', (), MAKE_PAIRS_CONFIG)()
    
    RNG.seed(args.seed)
    root = Path(args.data_root)
    disjoint_dir = root / "disjoint"
    images_dir = root / "images"
    categories_csv = root / "categories.csv"
    meta_path = root / "polyvore_item_metadata.json"

    sets = load_split_sets(disjoint_dir, args.split)
    cat_map = load_categories(categories_csv) if categories_csv.exists() else {}
    meta = load_item_metadata(meta_path) if meta_path.exists() else {}

    items_by_set, items_by_cat, item_img, item_cat = build_indices(sets, images_dir, meta, cat_map)
    allowed = parse_allowed_pairs(args.allowed_pairs)

    # Positives
    seen = set()
    pos = generate_positive_pairs(sets, item_cat, item_img, allowed, seen)

    # Negatives
    neg = generate_negative_pairs(pos, items_by_cat, args.neg_per_pos, args.hard_negatives)

    rows = pos + neg
    RNG.shuffle(rows)

    # Stats
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = len(rows) - n_pos
    pct = (100.0 * n_pos / len(rows)) if rows else 0.0
    cat_counts = Counter((r["a_cat"], r["b_cat"]) for r in rows if r["label"] == 1)
    print(f"[{args.split}] Total: {len(rows)} | Pos: {n_pos} | Neg: {n_neg} | {pct:.1f}% positives")
    print("Top positive cat pairs:", cat_counts.most_common(10))

    # Output
    out_path = Path("data") / "intermediate_pairs" / args.split / "pairs.jsonl"
    write_jsonl(rows, out_path)
    print(f"Wrote pairs → {out_path}")

    # Optional items list to drive segmentation later
    if args.items_out:
        uniq = []
        seen_items = set()
        for r in rows:
            for k in ("a_img","b_img"):
                if r[k] not in seen_items:
                    uniq.append(r[k]); seen_items.add(r[k])
        p = Path(args.items_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(uniq))
        print(f"Wrote unique item image paths → {p}")

if __name__ == "__main__":
    main()
