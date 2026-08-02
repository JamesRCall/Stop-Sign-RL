"""Generic training entry point.

The legacy ``train_single_stop_sign.py`` module remains the implementation and
checkpoint import path.  This wrapper gives new experiments a sign-agnostic CLI
without breaking old serialized policies.
"""

import runpy


if __name__ == "__main__":
    runpy.run_module("train_single_stop_sign", run_name="__main__")
