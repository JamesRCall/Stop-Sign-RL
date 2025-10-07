# utils/save_callbacks.py
import os, json
from typing import Optional, Dict, Any, Tuple, List
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


def _to_py(obj):
    """Recursively convert numpy / tensors / exotic types into JSON-serializable Python types."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if hasattr(obj, "__dict__"):
        try:
            return _to_py(vars(obj))
        except Exception:
            return str(obj)
    return str(obj)


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class SaveImprovingOverlaysCallback(BaseCallback):
    """
    Keep a leaderboard of the best overlays (size max_saved).
    When full, evict the current worst and insert the new one.

    Scoring rule:
      - mode == "adversary":   better = LOWER after_conf
                               tie-break: LOWER total_area_mask_frac, LOWER count
      - mode == "helper":      better = HIGHER after_conf
                               tie-break: LOWER total_area_mask_frac, LOWER count
    Threshold still applies as a gate (set to 0 to keep everything ranked).
    """

    def __init__(
        self,
        save_dir: str,
        threshold: float = 0.02,
        mode: str = "auto",           # "adversary" | "helper" | "auto" (read from info["objective"])
        max_saved: int = 300,
        verbose: int = 0,
        tb_callback: Optional[object] = None,   # TensorboardOverlayCallback or compatible
    ):
        super().__init__(verbose)
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.threshold = float(threshold)
        self.mode = mode
        self.max_saved = int(max_saved)
        self.tb_callback = tb_callback

        # leaderboard entries: dict with keys
        # {"key": tuple, "json_path": str, "png_path": Optional[str]}
        self._top: List[Dict[str, Any]] = []

    # ----------------- Scoring / ordering -----------------

    def _resolve_mode(self, info: Dict[str, Any]) -> str:
        if self.mode in ("adversary", "helper"):
            return self.mode
        return str(info.get("objective", "adversary")).lower()

    def _extract_metrics(self, info: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        Returns (base_conf, after_conf, total_area_mask_frac, count)
        with safe float coercion and defaults.
        """
        base_conf   = _safe_float(info.get("base_conf", 0.0))
        after_conf  = _safe_float(info.get("after_conf", info.get("conf_after", 0.0)))
        area_frac   = _safe_float(info.get("total_area_mask_frac", 0.0))
        params      = info.get("params", {}) or {}
        count       = _safe_float(params.get("count", 0))
        return base_conf, after_conf, area_frac, count

    def _passes_threshold(self, delta: float, mode: str) -> bool:
        if mode == "helper":
            return delta >= self.threshold
        # adversary: want negative delta (after < base)
        return (-delta) >= self.threshold

    def _make_key(self, mode: str, after_conf: float, area_frac: float, count: float) -> Tuple:
        """
        Lexicographic sort key:
          adversary:   smaller after_conf better; tie → smaller area → smaller count
          helper:      larger  after_conf better; tie → smaller area → smaller count
        We implement "best-first" sorting by using this key AND choosing min() for adversary,
        max() for helper to find the "worst" to evict.
        """
        if mode == "adversary":
            # Lower is better → key is (after_conf, area, count)
            return (after_conf, area_frac, count)
        else:
            # Higher is better → invert primary so that "larger is better" still sorts correctly
            # BUT for eviction we’ll use max() with the same key, see _find_worst_index.
            return (after_conf, -area_frac, -count)

    def _is_better(self, mode: str, key_a: Tuple, key_b: Tuple) -> bool:
        """
        True if A better than B per mode. We compare the raw metrics, not inverted.
        """
        if mode == "adversary":
            # smaller after_conf wins; then smaller area; then smaller count
            if key_a[0] != key_b[0]:
                return key_a[0] < key_b[0]
            if key_a[1] != key_b[1]:
                return key_a[1] < key_b[1]
            return key_a[2] < key_b[2]
        else:
            # helper: larger after_conf wins; tiebreaks same (smaller area/count)
            if key_a[0] != key_b[0]:
                return key_a[0] > key_b[0]
            if key_a[1] != key_b[1]:
                return key_a[1] < key_b[1]
            return key_a[2] < key_b[2]

    def _find_worst_index(self, mode: str) -> int:
        """
        Return index of the current worst entry in self._top.
        For adversary: worst = max by (after_conf, area, count)
        For helper:    worst = min by (after_conf, -area, -count)   (because "after_conf high is best")
        """
        if not self._top:
            return -1
        if mode == "adversary":
            # highest after_conf (then larger area, larger count) is worst
            worst_idx = max(range(len(self._top)), key=lambda i: self._top[i]["key"])
        else:
            # helper: lowest after_conf is worst given our key definition
            worst_idx = min(range(len(self._top)), key=lambda i: self._top[i]["key"])
        return worst_idx

    # ----------------- Saving / TB logging -----------------

    def _save_record(self, stem: str, info: Dict[str, Any], step: int, env_idx: int) -> Dict[str, Any]:
        """
        Save PNG and JSON; return dict with { 'json_path', 'png_path', 'key', 'meta' }.
        """
        # Build JSON meta and add png_path after saving image
        meta = {k: v for k, v in info.items() if k not in ("composited_pil",)}
        meta = _to_py(meta)
        meta["global_step"] = step
        meta["env_index"] = env_idx
        meta["png_path"] = None  # will fill below

        # Save image if present
        png_path = None
        img = info.get("composited_pil", None)
        if img is not None:
            try:
                png_path = os.path.join(self.save_dir, f"{stem}.png")
                img.save(png_path)
                meta["png_path"] = png_path
            except Exception as e:
                if self.verbose:
                    print(f"[save image failed] {stem}.png: {e}")

        # Save JSON
        json_path = os.path.join(self.save_dir, f"{stem}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.verbose:
                print(f"[save json failed] {stem}.json: {e}")

        # Compute key for ranking
        mode = self._resolve_mode(info)
        _, after_conf, area_frac, count = self._extract_metrics(info)
        key = self._make_key(mode, after_conf, area_frac, count)

        return {"json_path": json_path, "png_path": png_path, "key": key, "meta": meta}

    def _delete_entry_files(self, entry: Dict[str, Any]) -> None:
        for path in (entry.get("json_path"), entry.get("png_path")):
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    # ----------------- SB3 hook -----------------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        step = int(self.num_timesteps)

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            mode = self._resolve_mode(info)
            base_conf, after_conf, area_frac, count = self._extract_metrics(info)
            delta = _safe_float(info.get("delta_conf", after_conf - base_conf))

            # Gate by threshold (set to 0 to keep everything ranked)
            if not self._passes_threshold(delta, mode):
                continue

            # Build a stable filename stem: objective + step + env
            stem = f"{mode}_step{step:09d}_env{env_idx:02d}"
            new_entry = self._save_record(stem, info, step, env_idx)

            if len(self._top) < self.max_saved:
                # buffer not full: just append
                self._top.append(new_entry)
                if self.verbose:
                    print(f"[saved {len(self._top)}/{self.max_saved}] after_conf={after_conf:.3f} "
                          f"area={area_frac:.3f} count={int(count)} ({stem})")
            else:
                # buffer full: compare to current worst; if better, replace
                worst_idx = self._find_worst_index(mode)
                worst = self._top[worst_idx]
                is_better = self._is_better(mode, new_entry["key"], worst["key"])
                if is_better:
                    # delete files for old worst
                    self._delete_entry_files(worst)
                    # replace in leaderboard
                    self._top[worst_idx] = new_entry
                    if self.verbose:
                        print(f"[replaced worst] after_conf={after_conf:.3f} area={area_frac:.3f} "
                              f"count={int(count)} ({stem})")
                else:
                    # New one is worse → discard files we just wrote
                    self._delete_entry_files(new_entry)

            # Optional: log to TensorBoard
            if self.tb_callback is not None:
                try:
                    # Log the new record (or replaced record); TB callback will read png_path if present
                    self.tb_callback.log_overlay_record(new_entry["meta"], global_step=step)
                except Exception as e:
                    if self.verbose:
                        print("[Saver] TB log failed:", e)

        return True
