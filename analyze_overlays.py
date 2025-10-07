# analyze_overlays.py
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Point this at the folder you trained to (change if you ran helper)
OVERLAY_DIR = "./runs/overlays_adversary"   # was "./runs/overlays_single"

def load_logs(overlay_dir=OVERLAY_DIR):
    rows = []
    for jpath in sorted(glob.glob(os.path.join(overlay_dir, "overlay_*.json"))):
        with open(jpath, "r") as f:
            rec = json.load(f)
        p = rec.get("params", {})
        delta = float(rec.get("delta_conf", 0.0))
        objective = (rec.get("objective") or "helper").lower()  # "helper" or "adversary"
        # 2) Normalize so that higher == better in the run’s own direction
        effect_dir = delta if objective == "helper" else -delta

        base = {
            "file": os.path.basename(jpath),
            "objective": objective,
            "base_conf": rec.get("base_conf", 0.0),
            # some runs wrote "conf_after", others "after_conf" — handle both
            "conf_after": rec.get("conf_after", rec.get("after_conf", 0.0)),
            "delta_conf": delta,
            "effect_dir": effect_dir,  # use this for ranking/plots
            "count": p.get("count"),
            "size_scale": p.get("size_scale"),
            "alpha": p.get("alpha"),
            "mode": p.get("mode"),
            "color_mean_r": (p.get("color_mean") or [None,None,None])[0],
            "color_mean_g": (p.get("color_mean") or [None,None,None])[1],
            "color_mean_b": (p.get("color_mean") or [None,None,None])[2],
            "color_std": p.get("color_std")
        }
        rows.append(base)
    # Sort by effect in the correct direction for objective
    return pd.DataFrame(rows).sort_values("effect_dir", ascending=False)

def summarize_blob_overlays(overlay_dir=OVERLAY_DIR):
    b_rows = []
    for jpath in sorted(glob.glob(os.path.join(overlay_dir, "overlay_*.json"))):
        with open(jpath, "r") as f:
            rec = json.load(f)
        delta = float(rec.get("delta_conf", 0.0))
        objective = (rec.get("objective") or "helper").lower()
        effect_dir = delta if objective == "helper" else -delta
        params = rec.get("params", {})
        for blob in rec.get("overlays", []):
            b_rows.append({
                "file": os.path.basename(jpath),
                "objective": objective,
                "delta_conf": delta,
                "effect_dir": effect_dir,   # normalized sign
                "mode": blob.get("mode"),
                "area_frac": blob.get("area_frac"),
                "compactness": blob.get("compactness"),
                "alpha": blob.get("alpha"),
                "color_r": (blob.get("color") or [None,None,None])[0],
                "color_g": (blob.get("color") or [None,None,None])[1],
                "color_b": (blob.get("color") or [None,None,None])[2],
                "count": params.get("count"),
                "size_scale": params.get("size_scale")
            })
    return pd.DataFrame(b_rows).sort_values("effect_dir", ascending=False)

def main():
    if not os.path.isdir(OVERLAY_DIR):
        print(f"Directory not found: {OVERLAY_DIR}")
        return
    df = load_logs(OVERLAY_DIR)
    if df.empty:
        print("No JSON logs found. Train first, or lower the save threshold.")
        return

    print("\n=== TOP 10 overlay sets (direction-normalized) ===")
    print(df[["file","objective","effect_dir","delta_conf","conf_after","base_conf","count","alpha","size_scale","mode",
              "color_mean_r","color_mean_g","color_mean_b","color_std"]].head(10).to_string(index=False))

    print("\n=== Mode summary (mean effect_dir) ===")
    print(df.groupby("mode")["effect_dir"].agg(["mean","std","count"]).sort_values("mean", ascending=False))

    df["count_bin"] = pd.cut(df["count"], bins=[0,10,25,50,75,100],
                             labels=["1-10","11-25","26-50","51-75","76-100"])
    df["alpha_bin"] = pd.cut(df["alpha"], bins=[0,0.25,0.5,0.75,1.0],
                             labels=["0.10-0.25","0.25-0.50","0.50-0.75","0.75-0.90"])

    print("\n=== Count-bin summary (mean effect_dir) ===")
    print(df.groupby("count_bin")["effect_dir"].agg(["mean","std","count"]).sort_values("mean", ascending=False))

    print("\n=== Alpha-bin summary (mean effect_dir) ===")
    print(df.groupby("alpha_bin")["effect_dir"].agg(["mean","std","count"]).sort_values("mean", ascending=False))

    # Per-blob view
    bdf = summarize_blob_overlays(OVERLAY_DIR)
    if not bdf.empty:
        print("\n=== Per-blob descriptors vs effect_dir (corr) ===")
        corr = bdf[["effect_dir","area_frac","compactness","alpha"]].corr().loc[
            ["area_frac","compactness","alpha"], "effect_dir"]
        print(corr.sort_values(ascending=False))

        os.makedirs("./runs/analysis", exist_ok=True)

        # 1) effect_dir vs count
        plt.figure()
        plt.scatter(df["count"], df["effect_dir"], s=16, alpha=0.6)
        plt.xlabel("Blob count (set-level)")
        plt.ylabel("Effect (direction-normalized)")
        plt.title("Effect vs Blob Count")
        plt.grid(True, alpha=0.3)
        plt.savefig("./runs/analysis/effect_vs_count.png", dpi=160, bbox_inches="tight")

        # 2) effect_dir by mode (boxplot)
        plt.figure()
        df.boxplot(column="effect_dir", by="mode")
        plt.title("Effect by mode")
        plt.suptitle("")
        plt.ylabel("Effect (direction-normalized)")
        plt.savefig("./runs/analysis/effect_by_mode.png", dpi=160, bbox_inches="tight")

        # 3) Per-blob area vs effect_dir
        plt.figure()
        plt.scatter(bdf["area_frac"], bdf["effect_dir"], s=6, alpha=0.4)
        plt.xlabel("Blob area fraction")
        plt.ylabel("Effect (set-level, normalized)")
        plt.title("Per-blob Area vs Effect")
        plt.grid(True, alpha=0.3)
        plt.savefig("./runs/analysis/effect_vs_areafrac.png", dpi=160, bbox_inches="tight")

        print("\nSaved plots in ./runs/analysis/")
    else:
        print("\nNo per-blob metadata found (did any overlays get saved?).")

if __name__ == "__main__":
    main()
