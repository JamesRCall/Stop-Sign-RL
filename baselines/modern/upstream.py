"""Download hash-pinned author code; execute only audited optimizer functions.

No upstream dataset/model loading or top-level experiment code is executed.
The original repositories are downloaded on the user's machine, not vendored.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import types
import urllib.request

SOURCES = {
    "fipatch_attack.py": {
        "url": "https://anonymous.4open.science/api/repo/FIPatch-FA52/file/fipatch_attack.py",
        "sha256": "e32b0cf35ca7c30105be0d6a4fb8b44abf42cf4f3c46b8ffcd2dda5afcae2d76",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/8e608f20d2bc14ffe312635285e0125c-Abstract-Conference.html",
        "year": 2025,
    },
    "impact_DE.py": {
        "url": "https://raw.githubusercontent.com/yangzh216/IMPACT/962cf92f26745e823a4abe3c664f9b414cafcce3/DE.py",
        "sha256": "dbc838a7c879b0aca14b5d56f26a65113384da491da9620e36a6589c0c855ca1",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/8172ca14a9ec80a3409113d1d1f8bc42-Abstract-Conference.html",
        "revision": "962cf92f26745e823a4abe3c664f9b414cafcce3",
        "year": 2025,
    },
}


def fetch_sources(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        path = directory / name
        if not path.exists():
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "DIRP-baseline-research"})
            with urllib.request.urlopen(request, timeout=40) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise RuntimeError(f"Upstream source changed: {name}; refusing unverified code")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {path}")
    return SOURCES


def load_kernels(directory: Path):
    import numpy as np
    from pyswarm import pso
    from sklearn.cluster import KMeans

    fetch_sources(directory)
    allowed = {
        "fipatch_attack.py": {"n_random_restart_pso"},
        "impact_DE.py": {"init_population", "mutation", "crossover", "fix_patch_num",
                         "move_towards_nearby_point", "squeeze_individual"},
    }
    loaded = []
    for name, names in allowed.items():
        path = directory / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise RuntimeError(f"Expected optimizer functions absent: {name}")
        module = types.ModuleType(name.replace(".py", ""))
        module.__dict__.update(np=np, pso=pso, KMeans=KMeans, F=2, CR=0.8)
        code = compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec")
        exec(code, module.__dict__)
        loaded.append(module)
    return tuple(loaded)
