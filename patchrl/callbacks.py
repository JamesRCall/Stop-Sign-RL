"""Query-budgeted training callbacks with resumable, auditable checkpoints."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class QueryBudgetCallback(BaseCallback):
    """Stop by detector-image queries and checkpoint on the same unit.

    PPO timesteps are not comparable when an action can buy a different number
    of detector images.  This callback makes the detector-image ledger the hard
    training budget and records the best dense potential and terminal success
    rate without introducing a moving dual penalty into the reward.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        maximum_queries: int,
        checkpoint_every_queries: int,
        no_progress_after_queries: int,
        minimum_progress: float,
        success_window: int = 32,
        required_success_rate: float = 0.80,
        initial_queries: int = 0,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.maximum_queries = int(maximum_queries)
        self.checkpoint_every_queries = int(checkpoint_every_queries)
        self.no_progress_after_queries = int(no_progress_after_queries)
        self.minimum_progress = float(minimum_progress)
        self.success_window = int(success_window)
        self.required_success_rate = float(required_success_rate)
        self.initial_queries = int(initial_queries)
        if self.maximum_queries < 1:
            raise ValueError("maximum_queries must be positive")
        if not 0 <= self.initial_queries < self.maximum_queries:
            raise ValueError("initial_queries must be in [0, maximum_queries)")
        if self.checkpoint_every_queries < 1:
            raise ValueError("checkpoint_every_queries must be positive")
        if self.no_progress_after_queries < 0:
            raise ValueError("no_progress_after_queries must be non-negative")
        if not 0.0 <= self.minimum_progress <= 1.0:
            raise ValueError("minimum_progress must be in [0, 1]")
        if self.success_window < 1:
            raise ValueError("success_window must be positive")
        if not 0.0 <= self.required_success_rate <= 1.0:
            raise ValueError("required_success_rate must be in [0, 1]")

        self.next_checkpoint = (
            (self.initial_queries // self.checkpoint_every_queries) + 1
        ) * self.checkpoint_every_queries
        self.best_potential = 0.0
        # Zero is the clean reset baseline and is therefore always attainable.
        # A negative edit must never become the run's "best" progress.
        self.best_progress = 0.0
        self.terminal_successes: deque[bool] = deque(maxlen=self.success_window)
        self.stop_reason = "maximum_query_budget"
        self.final_queries = 0

    def _query_count(self) -> int:
        values = self.training_env.get_attr("lifetime_detector_queries")
        return self.initial_queries + int(sum(int(value) for value in values))

    def _record_infos(self) -> None:
        infos = list(self.locals.get("infos", []) or [])
        dones = list(self.locals.get("dones", []) or [])
        for index, info in enumerate(infos):
            def scalar(name: str, lower: float, upper: float) -> float:
                if name not in info:
                    raise RuntimeError(
                        f"MetaPatchEnv callback info is missing scalar field {name!r}"
                    )
                value = info[name]
                if isinstance(value, bool):
                    raise RuntimeError(f"callback info field {name!r} must be numeric")
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"callback info field {name!r} must be numeric"
                    ) from exc
                if not math.isfinite(numeric) or not lower <= numeric <= upper:
                    raise RuntimeError(
                        f"callback info field {name!r} must be finite and in "
                        f"[{lower}, {upper}]"
                    )
                return numeric

            initial = scalar("initial_dense_potential", 0.0, 1.0)
            potential = scalar("dense_potential", 0.0, 1.0)
            progress = scalar("dense_progress", -1.0, 1.0)
            expected_progress = potential - initial
            if not math.isclose(progress, expected_progress, abs_tol=1e-9):
                raise RuntimeError(
                    "dense_progress must equal dense_potential minus "
                    "initial_dense_potential"
                )
            self.best_potential = max(self.best_potential, initial, potential)
            self.best_progress = max(self.best_progress, progress)
            if index < len(dones) and bool(dones[index]):
                self.terminal_successes.append(bool(info.get("attack_success", False)))

    @property
    def rolling_success_rate(self) -> float | None:
        if not self.terminal_successes:
            return None
        return float(sum(self.terminal_successes) / len(self.terminal_successes))

    def _state(self, queries: int) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "updated_utc": _utc_now(),
            "detector_image_queries": int(queries),
            "policy_timesteps": int(self.num_timesteps),
            "maximum_detector_image_queries": self.maximum_queries,
            "best_dense_potential": float(self.best_potential),
            "best_dense_progress": float(self.best_progress),
            "rolling_terminal_success_rate": self.rolling_success_rate,
            "rolling_window_size": len(self.terminal_successes),
            "stop_reason": self.stop_reason,
        }

    def _checkpoint(self, queries: int) -> None:
        directory = self.output_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"policy_q{int(queries):09d}"
        self.model.save(str(stem))
        _atomic_json(self.output_dir / "training_state.json", self._state(queries))

    def _on_step(self) -> bool:
        self._record_infos()
        queries = self._query_count()
        self.final_queries = queries
        if queries >= self.next_checkpoint:
            self._checkpoint(queries)
            while self.next_checkpoint <= queries:
                self.next_checkpoint += self.checkpoint_every_queries

        if (
            len(self.terminal_successes) == self.success_window
            and float(self.rolling_success_rate or 0.0) >= self.required_success_rate
        ):
            self.stop_reason = "rolling_success_gate"
            return False
        if (
            self.no_progress_after_queries > 0
            and queries >= self.no_progress_after_queries
            and self.best_progress + 1e-12 < self.minimum_progress
        ):
            self.stop_reason = "no_progress_gate"
            return False
        if queries >= self.maximum_queries:
            self.stop_reason = "maximum_query_budget"
            return False
        return True

    def _on_training_end(self) -> None:
        self.final_queries = self._query_count()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.output_dir / "policy_final"))
        _atomic_json(
            self.output_dir / "training_state.json",
            self._state(self.final_queries),
        )


__all__ = ["QueryBudgetCallback"]
