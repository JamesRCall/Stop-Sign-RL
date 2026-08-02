"""Generic public entry point for the traffic-sign patch environment.

The implementation remains in ``stop_sign_grid_env`` so existing checkpoints and
third-party imports do not break.  New code should import ``TrafficSignGridEnv``
from this module; ``StopSignGridEnv`` is retained only as a compatibility alias.
"""

from envs.stop_sign_grid_env import StopSignGridEnv, TrafficSignGridEnv

__all__ = ["TrafficSignGridEnv", "StopSignGridEnv"]
