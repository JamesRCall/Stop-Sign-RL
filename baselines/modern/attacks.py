"""Detector/UV adaptations; revision 4 fixes IMPACT edge-cell eligibility.

Author optimizer kernels are retained. Material warm starts, sign-support
repair, variable circle radius, and tile-space ES are explicit adaptations.
No successful DIRP/Greedy/Random masks or detector gradients are used.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw


class _PhaseComplete(Exception):
    pass


def circle_patch(shape, sign_mask, centers, radius, colors):
    """Rasterize circles with their centers repaired onto the sign support."""
    h, w = shape
    valid = np.argwhere(sign_mask)
    if not len(valid):
        raise ValueError("Sign support is empty")
    mask_image = Image.new("L", (w, h), 0)
    rgb_image = Image.new("RGB", (w, h), 0)
    draw, rgb_draw = ImageDraw.Draw(mask_image), ImageDraw.Draw(rgb_image)
    for (x, y), color in zip(centers, colors):
        x, y = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        if not sign_mask[y, x]:
            y, x = valid[np.argmin((valid[:, 0]-y)**2 + (valid[:, 1]-x)**2)]
        bounds = (int(x)-radius, int(y)-radius, int(x)+radius, int(y)+radius)
        draw.ellipse(bounds, fill=255)
        rgb_draw.ellipse(bounds, fill=tuple(map(int, color)))
    mask = (np.asarray(mask_image) > 0) & sign_mask
    if not mask.any():
        raise RuntimeError("Circle repair produced an empty patch")
    return mask, np.asarray(rgb_image, dtype=np.float32)


def search_score(row):
    return row.get("optimization_score", row["score"])


def _settings(oracle):
    config = getattr(oracle, "config", {})
    return config.get("color_mode", "rgb"), np.asarray(
        getattr(oracle, "material_rgb", (230, 255, 0)), dtype=np.float32)


def fipatch(oracle, upstream, seed: int, *, circles: int = 3, swarm: int = 16):
    """Material-seeded circle PSO, followed by joint RGB/geometry refinement.

    Geometry gets 60% of a full-RGB run's budget. Full RGB is reachable in
    phase 2. Material-only mode spends all queries on placement/radius.
    The native random-restart PSO kernel is used throughout.
    """
    np.random.seed(seed)
    h, w = oracle.shape
    color_mode, material = _settings(oracle)
    radius_target = math.sqrt(oracle.area_target * oracle.sign_pixels / (circles * math.pi))
    levels = getattr(oracle, "config", {}).get("search_area_levels", [.25, .5])
    radius_low = max(1.0, .5*radius_target)
    radius_high = max(2.0, math.sqrt(max(levels)*oracle.sign_pixels/(circles*math.pi)))
    low = [0.0, 0.0]*circles + [radius_low]
    high = [float(w-1), float(h-1)]*circles + [radius_high]
    phase_end = (max(1, int(.6*oracle.candidate_limit)) if color_mode == "rgb"
                 else oracle.candidate_limit)
    best_params, best_score = None, -np.inf
    oracle.search_stage = "material_geometry_pso"

    def geometry_objective(params):
        nonlocal best_params, best_score
        if oracle.evaluations >= phase_end:
            raise _PhaseComplete()
        centers = np.asarray(params[:2*circles]).reshape(-1, 2)
        mask, rgb = circle_patch(oracle.shape, oracle.sign_mask, centers,
            int(params[2*circles]), np.tile(material, (circles, 1)))
        score = search_score(oracle.evaluate(mask, rgb))
        if score > best_score:
            best_params, best_score = np.array(params, copy=True), score
        return -score

    try:
        while True:
            upstream.n_random_restart_pso(geometry_objective, low, high,
                swarmsize=swarm, maxiter=40, n_restarts=1)
    except _PhaseComplete:
        pass
    if color_mode == "material":
        oracle.evaluate(*circle_patch(oracle.shape, oracle.sign_mask,
            best_params[:2*circles].reshape(-1, 2), int(best_params[2*circles]),
            np.tile(material, (circles, 1))))
        return
    oracle.search_stage = "joint_rgb_geometry_pso"
    span = np.tile([.25*w, .25*h], circles)
    center_low = np.maximum(0, best_params[:2*circles]-span)
    center_high = np.minimum(np.tile([w-1, h-1], circles), best_params[:2*circles]+span)
    low = list(center_low) + [radius_low] + [-255.0]*(3*circles)
    high = list(center_high) + [radius_high] + [255.0]*(3*circles)

    def joint_objective(params):
        centers = np.asarray(params[:2*circles]).reshape(-1, 2)
        # Full RGB is reachable; clipping also preserves material-color starts.
        colors = np.clip(material + np.asarray(params[2*circles+1:]).reshape(-1, 3), 0, 255)
        mask, rgb = circle_patch(oracle.shape, oracle.sign_mask, centers,
            int(params[2*circles]), colors)
        return -search_score(oracle.evaluate(mask, rgb))

    while True:
        upstream.n_random_restart_pso(joint_objective, low, high,
            swarmsize=swarm, maxiter=40, n_restarts=1)


def tile_support(sign_mask, tile, coverage_threshold=.6):
    """Use DIRP's coverage fraction of the ACTUAL cropped cell, including edges."""
    h, w = sign_mask.shape
    side = math.ceil(max(h, w)/tile)
    areas = np.zeros(side*side, dtype=int)
    sizes = np.zeros(side*side, dtype=int)
    for index in range(side*side):
        r, c = divmod(index, side)
        cell = sign_mask[r*tile:(r+1)*tile, c*tile:(c+1)*tile]
        areas[index], sizes[index] = int(cell.sum()), cell.size
    coverage = np.divide(areas, sizes, out=np.zeros_like(areas, dtype=float), where=sizes>0)
    valid = np.flatnonzero((sizes > 0) & (coverage >= coverage_threshold))
    if len(valid) < 4:
        raise ValueError("Too few nonempty sign cells for IMPACT")
    return side, valid, areas


def repair_tiles(code, colors, valid, side):
    """Move off-sign tiles to distinct legal cells, retaining color association."""
    selected = np.flatnonzero(code)
    if len(selected) > len(valid):
        raise ValueError("More selected tiles than legal cells")
    valid_set = set(map(int, valid))
    occupied = set(map(int, selected)) & valid_set
    available = np.array(sorted(valid_set-occupied), dtype=int)
    mapped = []
    for index in selected:
        if int(index) in valid_set:
            mapped.append(int(index)); continue
        r, c = divmod(int(index), side)
        nearest = int(np.argmin((available//side-r)**2 + (available%side-c)**2))
        mapped.append(int(available[nearest]))
        available = np.delete(available, nearest)
    order = np.argsort(mapped)
    repaired = np.zeros_like(code)
    repaired[np.asarray(mapped, dtype=int)] = 1
    return repaired, np.asarray(colors)[order].copy()


def impact(oracle, upstream, seed: int, *, population: int = 16, phase1_fraction: float = .7):
    """Native mask/content DE, then fixed-mask, tile-space (1+1)-ES.

    Starts mix known material, nearby ink, unrestricted RGB and author colors.
    ES perturbs rendered tiles instead of ~200k unrelated pixels. Material-only
    mode keeps content fixed and spends the whole budget on native DE placement.
    """
    np.random.seed(seed)
    h, w = oracle.shape
    color_mode, material = _settings(oracle)
    tile = oracle.grid_cell
    threshold = getattr(oracle, "config", {}).get("cell_cover_thresh", .6)
    side, valid, areas = tile_support(oracle.sign_mask, tile, threshold)
    if hasattr(oracle, "env"):
        coordinates = np.argwhere(oracle.env._valid_cells)
        expected = coordinates[:, 0]*side + coordinates[:, 1]
        if not np.array_equal(valid, expected):
            raise RuntimeError("IMPACT eligible cells differ from the DIRP environment")
    dimension = side*side
    levels = getattr(oracle, "config", {}).get("search_area_levels", [.25, .5])
    # The old area target is a penalty, not a hard 25% support constraint.
    # Native kernels require fixed cardinality, so independent DE populations
    # explore predefined support sizes within one shared query budget.
    counts = sorted(set(max(3, min(len(valid)-1,
        round(level*oracle.sign_pixels/np.median(areas[valid])))) for level in levels))
    per_island = max(4, min(population//len(counts), oracle.candidate_limit//(8*len(counts))))
    islands = []
    for tiles in counts:
        clusters = min(3, tiles)
        masks, colors = upstream.init_population(per_island, clusters, tiles, side*tile, tile)
        for i in range(per_island):
            masks[i], colors[i] = repair_tiles(masks[i], colors[i], valid, side)
            if color_mode == "material" or i % 4 == 0:
                colors[i][:] = material
            elif i % 4 == 1:
                colors[i][:] = np.clip(material + np.random.normal(0, 48, colors[i].shape), 0, 255)
            elif i % 4 == 2:
                colors[i][:] = np.random.uniform(0, 255, colors[i].shape)
        islands.append(dict(masks=masks, colors=colors, tiles=tiles, clusters=clusters))

    def decode(mask_code, color_code):
        mask = np.zeros((side*tile, side*tile), dtype=bool)
        rgb = np.zeros((side*tile, side*tile, 3), dtype=np.float32)
        for i, index in enumerate(np.flatnonzero(mask_code)):
            row, col = divmod(int(index), side)
            sl = np.s_[row*tile:(row+1)*tile, col*tile:(col+1)*tile]
            mask[sl] = True
            rgb[sl] = color_code[i]
        return mask[:h, :w], rgb[:h, :w]

    oracle.search_stage = "mask_content_de" if color_mode == "rgb" else "material_mask_de"
    for island in islands:
        island["fitness"] = np.asarray([search_score(oracle.evaluate(*decode(mask, color)))
            for mask, color in zip(island["masks"], island["colors"])])
    phase1_limit = (max(per_island*len(islands), int(oracle.candidate_limit*phase1_fraction))
                    if color_mode == "rgb" else oracle.candidate_limit)
    while oracle.evaluations < phase1_limit:
        for island in islands:
            if oracle.evaluations >= phase1_limit:
                break
            masks, colors, fitness = island["masks"], island["colors"], island["fitness"]
            mutant = upstream.mutation(per_island, [masks, colors])
            for i in range(per_island):
                if not np.any(mutant[0][i]):
                    mutant[0][i] = masks[i].copy()
            trials = upstream.crossover(per_island, mutant, [masks, colors],
                island["clusters"], island["tiles"], dimension)
            for i in range(per_island):
                if oracle.evaluations >= phase1_limit:
                    break
                trials[0][i], trials[1][i] = repair_tiles(trials[0][i], trials[1][i], valid, side)
                if color_mode == "material":
                    trials[1][i][:] = material
                score = search_score(oracle.evaluate(*decode(trials[0][i], trials[1][i])))
                if score > fitness[i]:
                    masks[i], colors[i], fitness[i] = trials[0][i], trials[1][i], score
    island = max(islands, key=lambda state: float(np.max(state["fitness"])))
    masks, colors, fitness, tiles = island["masks"], island["colors"], island["fitness"], island["tiles"]
    index = int(np.argmax(fitness))
    mask_code, texture = masks[index].copy(), colors[index].astype(np.float32).copy()
    score = float(fitness[index])
    if color_mode == "material":
        oracle.evaluate(*decode(mask_code, texture))
        return
    oracle.search_stage = "tile_content_es"
    palette = np.asarray([material, [0, 0, 0], [255, 255, 255], [255, 0, 0],
                          [0, 255, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255]], dtype=np.float32)
    iteration = 0
    while True:
        remaining_fraction = max(0, (oracle.candidate_limit-oracle.evaluations) /
            max(1, oracle.candidate_limit-phase1_limit))
        sigma = 8 + 56*remaining_fraction
        proposal = texture.copy()
        if iteration % 16 == 0:
            proposal[:] = palette[(iteration//16) % len(palette)]
        else:
            count = [1, max(1, tiles//4), tiles][iteration % 3]
            chosen = np.random.choice(tiles, count, replace=False)
            proposal[chosen] += np.random.normal(0, sigma, (1, 3))
        proposal = np.clip(proposal, 0, 255)
        result = oracle.evaluate(*decode(mask_code, proposal))
        if search_score(result) > score:
            texture, score = proposal, search_score(result)
        iteration += 1
