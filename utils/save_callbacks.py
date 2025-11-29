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
    Keep only the top-K overlays (max_saved, default 50).
    When full, evict the current worst and insert the new better one.

    Scoring:
      • mode == "adversary": smaller after_conf is better
      • mode == "helper":    larger after_conf is better
    """

    def __init__(
        self,
        save_dir: str,
        threshold: float = 0.0,
        mode: str = "auto",  # "adversary" | "helper" | "auto"
        max_saved: int = 50,
        verbose: int = 0,
        tb_callback: Optional[object] = None,
    ):
        super().__init__(verbose)
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.threshold = float(threshold)
        self.mode = mode
        self.max_saved = int(max_saved)
        self.tb_callback = tb_callback
        self._top: List[Dict[str, Any]] = []  # leaderboard

        # trace log file (append-only)
        self._trace_path = os.path.join(self.save_dir, "traces.ndjson")

    # ---------- scoring utilities ----------
    def _resolve_mode(self, info: Dict[str, Any]) -> str:
        if self.mode in ("adversary", "helper"):
            return self.mode
        return str(info.get("objective", "adversary")).lower()

    def _extract_metrics(self, info: Dict[str, Any]) -> Tuple[float, float, float, float]:
        base_conf = _safe_float(info.get("base_conf", 0.0))
        after_conf = _safe_float(info.get("after_conf", info.get("conf_after", 0.0)))
        area_frac = _safe_float(info.get("total_area_mask_frac", 0.0))
        count = _safe_float(info.get("params", {}).get("count", 0))
        return base_conf, after_conf, area_frac, count

    def _passes_threshold(self, delta: float, mode: str) -> bool:
        if mode == "helper":
            return delta >= self.threshold
        return (-delta) >= self.threshold  # adversary

    def _make_key(self, mode: str, after_conf: float, area_frac: float, count: float):
        # Lexicographic key for sorting / comparison
        if mode == "adversary":
            return (after_conf, area_frac, count)
        else:
            return (after_conf, -area_frac, -count)

    def _is_better(self, mode: str, key_a, key_b) -> bool:
        if mode == "adversary":
            # smaller after_conf wins
            if key_a[0] != key_b[0]:
                return key_a[0] < key_b[0]
            if key_a[1] != key_b[1]:
                return key_a[1] < key_b[1]
            return key_a[2] < key_b[2]
        else:
            # helper: larger after_conf wins
            if key_a[0] != key_b[0]:
                return key_a[0] > key_b[0]
            if key_a[1] != key_b[1]:
                return key_a[1] < key_b[1]
            return key_a[2] < key_b[2]

    def _find_worst_index(self, mode: str) -> int:
        if not self._top:
            return -1
        if mode == "adversary":
            return max(range(len(self._top)), key=lambda i: self._top[i]["key"])
        else:
            return min(range(len(self._top)), key=lambda i: self._top[i]["key"])

    # ---------- file helpers ----------
    def _delete_entry_files(self, entry: Dict[str, Any]):
        for p in (entry.get("json_path"), entry.get("png_path")):
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def _write_trace_line(self, info: Dict[str, Any], score: float):
        """Append a minimal JSON line with only the trace info."""
        trace = info.get("trace", {}) or {}
        mini = {
            "step": int(self.num_timesteps),
            "score": float(score),
            **_to_py(trace),
        }
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(mini, ensure_ascii=False) + "\n")
        except Exception as e:
            if self.verbose:
                print("[Saver] trace write failed:", e)

    def _save_record(self, stem: str, info: Dict[str, Any]) -> Dict[str, Any]:
        meta = {k: v for k, v in info.items() if k != "composited_pil"}
        meta = _to_py(meta)
        meta["global_step"] = int(self.num_timesteps)
        png_path = None
        img = info.get("composited_pil")
        if img is not None:
            try:
                png_path = os.path.join(self.save_dir, f"{stem}.png")
                img.save(png_path)
                meta["png_path"] = png_path
            except Exception as e:
                if self.verbose:
                    print(f"[save image failed] {stem}.png: {e}")
        json_path = os.path.join(self.save_dir, f"{stem}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.verbose:
                print(f"[save json failed] {stem}.json: {e}")
        mode = self._resolve_mode(info)
        _, after_conf, area_frac, count = self._extract_metrics(info)
        key = self._make_key(mode, after_conf, area_frac, count)
        return {"json_path": json_path, "png_path": png_path, "key": key, "meta": meta}

    # ---------- SB3 hook ----------
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            mode = self._resolve_mode(info)
            base_conf, after_conf, area_frac, count = self._extract_metrics(info)
            delta = _safe_float(info.get("delta_conf", after_conf - base_conf))
            score = base_conf - after_conf if mode == "adversary" else after_conf - base_conf

            if not self._passes_threshold(delta, mode):
                continue

            stem = f"{mode}_step{self.num_timesteps:09d}_env{env_idx:02d}"
            new_entry = self._save_record(stem, info)

            if len(self._top) < self.max_saved:
                self._top.append(new_entry)
                if self.verbose:
                    print(f"[saved {len(self._top)}/{self.max_saved}] {stem}")
            else:
                worst_idx = self._find_worst_index(mode)
                worst = self._top[worst_idx]
                if self._is_better(mode, new_entry["key"], worst["key"]):
                    self._delete_entry_files(worst)
                    self._top[worst_idx] = new_entry
                    if self.verbose:
                        print(f"[replaced worst] {stem}")
                else:
                    self._delete_entry_files(new_entry)

            # --- trace-only line ---
            self._write_trace_line(info, score)

            # --- TensorBoard ---
            if self.tb_callback is not None:
                try:
                    self.tb_callback.log_overlay_record(new_entry["meta"], global_step=self.num_timesteps)
                except Exception as e:
                    if self.verbose:
                        print("[Saver] TB log failed:", e)

        return True
