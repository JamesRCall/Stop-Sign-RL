"""Manual step-through of the grid env with image dumps for debugging."""
import os, glob, argparse
from PIL import Image, ImageDraw
import numpy as np

from envs.stop_sign_grid_env import StopSignGridEnv
from utils.uv_paint import VIOLET_GLOW

def load_bgs(folder):
    """
    Load a small set of background images for debugging.

    @param folder: Background folder.
    @return: List of PIL images.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    return [Image.open(p).convert("RGB") for p in paths][:20]

def main():
    """
    Run a short random policy roll-out and save intermediate frames.

    @return: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--bgdir", default="./data/backgrounds")
    p.add_argument("--yolo", default="./weights/yolo11n.pt")
    p.add_argument("--out", default="./_debug_grid")
    p.add_argument("--grid-cell", type=int, default=2, choices=[2,4])
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--eval-K", type=int, default=10)
    p.add_argument("--uv-threshold", type=float, default=0.70)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    stop_day = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    stop_uv_p = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv  = Image.open(stop_uv_p).convert("RGBA") if os.path.exists(stop_uv_p) else stop_day.copy()
    pole     = Image.open(os.path.join(args.data, "pole.png")).convert("RGBA")
    bgs      = load_bgs(args.bgdir)

    env = StopSignGridEnv(
        stop_sign_image=stop_day,
        stop_sign_uv_image=stop_uv,
        background_images=bgs,
        pole_image=pole,
        yolo_weights=args.yolo,
        yolo_device=args.device,
        img_size=(640,640),

        steps_per_episode=7000,
        eval_K=args.eval_K,
        grid_cell_px=args.grid_cell,
        max_cells=None,

        uv_paint=VIOLET_GLOW,      # single color pair
        use_single_color=True,
        uv_drop_threshold=args.uv_threshold,
        day_tolerance=0.05,
        lambda_day=1.0,
    )

    obs, info = env.reset()
    Image.fromarray(obs).save(os.path.join(args.out, "t00_obs_plain_day.png"))

    rng = np.random.default_rng(123)
    for t in range(1, args.steps + 1):
        a = rng.uniform(-1, 1, size=(2,)).astype(np.float32)  # (row, col)
        obs, rew, term, trunc, info = env.step(a)

        # Save the daylight observation (transform #0)
        Image.fromarray(obs).save(os.path.join(args.out, f"t{t:02d}_obs_day.png"))

        # Save the UV-on preview (this is what YOLO was scored on for preview)
        if "composited_pil" in info and info["composited_pil"] is not None:
            info["composited_pil"].save(os.path.join(args.out, f"t{t:02d}_uv_on.png"))

        # Save the current grid mask (just the trace, on sign coords)
        mask = Image.new("L", env.sign_rgba_day.size, 0)
        mdraw = ImageDraw.Draw(mask)
        Gw = env.Gw
        on = np.argwhere(env._episode_cells)
        for r, c in on:
            x0, y0, x1, y1 = env._cell_rects[r * Gw + c]
            mdraw.rectangle([x0, y0, x1, y1], fill=255)
        mask.save(os.path.join(args.out, f"t{t:02d}_grid_mask.png"))

        print(f"[t={t}] drop_on={info['drop_on']:.3f}  drop_day={info['drop_day']:.3f}  "
              f"reward={info['reward']:.3f}  cells={info['selected_cells']}")

        if term or trunc:
            print("Episode ended.")
            break

if __name__ == "__main__":
    main()
