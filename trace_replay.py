# trace_replay.py
import os, json, argparse
import numpy as np
from PIL import Image

# uses the same blob drawer as training
from envs.random_blobs import draw_randomized_blobs_set

def load_traces(ndjson_path: str):
    rows = []
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No rows found in: {ndjson_path}")
    return rows

def pick_trace(rows, index=None, step=None):
    if index is not None:
        if not (0 <= index < len(rows)):
            raise IndexError(f"--index {index} out of range (0..{len(rows)-1})")
        return rows[index]
    if step is not None:
        for r in rows:
            if int(r.get("step", -1)) == int(step):
                return r
        raise ValueError(f"No trace with step={step} found.")
    return rows[-1]  # default: last row

def main():
    ap = argparse.ArgumentParser(description="Export just the blob pattern (mask only, pre-transform) from a saved trace.")
    ap.add_argument("--traces", default="runs/overlays/traces.ndjson", help="Path to traces.ndjson")
    ap.add_argument("--index", type=int, default=None, help="Row index in NDJSON (0-based)")
    ap.add_argument("--step", type=int, default=None, help="Match a specific global step")
    ap.add_argument("--sign", default="data/stop_sign.png", help="Sign PNG with alpha (used only to clip mask to sign shape)")
    ap.add_argument("--outdir", default="replay_out", help="Where to save the exported mask")
    ap.add_argument("--w", type=int, default=640, help="Canvas width for mask export")
    ap.add_argument("--h", type=int, default=640, help="Canvas height for mask export")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) pick the trace row
    rows = load_traces(args.traces)
    tr = pick_trace(rows, args.index, args.step)

    # 2) pull the blob params from the trace
    pattern_seed = int(tr["pattern_seed"])
    count       = int(tr["count"])
    size_scale  = float(tr["size_scale"])
    # color_idx/alpha are irrelevant for a geometry-only mask

    # 3) load sign alpha to limit drawing region (so blobs stay on the sign)
    sign_rgba = Image.open(args.sign).convert("RGBA")
    sign_a    = sign_rgba.split()[-1]  # allowed mask

    # 4) create a blank RGB canvas the same size as the sign mask
    W, H = sign_rgba.size
    base_rgb = Image.new("RGB", (W, H), (0, 0, 0))

    # 5) re-create geometry deterministically using the saved seed
    rng = np.random.default_rng(pattern_seed)

    # IMPORTANT:
    # we want a mask of the blob shapes ONLY, so render on a black base with
    # WHITE blobs and alpha=1.0, then threshold to a binary mask.
    comp_rgb, _ = draw_randomized_blobs_set(
        base_pil=base_rgb,
        count=count,
        size_scale=size_scale,
        alpha=1.0,                      # fully opaque to generate a clear mask
        color_mean=(255, 255, 255),     # white blobs on black background
        color_std=0.0,
        mode="superellipse",            # must match training mode
        rng=rng,
        allowed_mask=sign_a,            # clip to sign shape like training
        area_cap=0.20,
        cap_relative_to_mask=True,
        single_color=True,
    )

    # 6) extract blob mask: comp_rgb is black where no blob, white where blob
    arr = np.array(comp_rgb, dtype=np.uint8)
    # turn any non-black pixel to white (blob); keep black as background
    blob_mask = ((arr[..., 0] > 0) | (arr[..., 1] > 0) | (arr[..., 2] > 0)).astype(np.uint8) * 255
    blob_mask_img = Image.fromarray(blob_mask, mode="L")

    # 7) (optional) place mask on a canvas you want (default exports at sign size).
    # If you want a different canvas (e.g., 640x640), center it:
    if (W, H) != (args.w, args.h):
        canvas = Image.new("L", (args.w, args.h), 0)
        x = (args.w - W) // 2
        y = (args.h - H) // 2
        canvas.paste(blob_mask_img, (x, y))
        blob_mask_img = canvas

    # 8) Save JUST the mask, pre-transform (no sign, no pole, no background)
    phase = tr.get("phase", "B")
    step  = tr.get("step", "NA")
    out_path = os.path.join(args.outdir, f"trace_mask_pre_{phase}_step{step}.png")
    blob_mask_img.save(out_path)
    print(f"✅ Saved pre-transform blob mask => {out_path}")

if __name__ == "__main__":
    main()
