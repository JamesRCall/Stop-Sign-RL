"""FIPatch circle PSO and IMPACT DE/ES, adapted to the common UV renderer.

These retain the author optimizer kernels. Classifier logits/losses, datasets,
and native image formation are replaced by the DIRP detector-score objective.
IMPACT uses the benchmark's 16-pixel tiles instead of its original 4-pixel tiles.
The modern methods optimize active RGB; the reused fixed-yellow baselines do not.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw


def fipatch(oracle, upstream, seed: int, *, circles: int = 1, swarm: int = 10):
    """Author restart PSO: shared radius, positions and one RGB per circle.

    Author default radius 7 on a 32px sign is scaled to the benchmark sign.
    Activation alpha comes from the shared UV material instead of being fitted
    independently, so the two methods use the same activation model.
    """
    np.random.seed(seed)
    h, w = oracle.shape
    scale = min(h, w) / 32.0
    low = [0.0, 0.0]*circles + [6.9*scale] + [0.0]*(3*circles)
    high = [float(w-1), float(h-1)]*circles + [7.1*scale] + [255.0]*(3*circles)

    def objective(params):
        mask_image = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_image)
        rgb_image = Image.new("RGB", (w, h), 0)
        rgb_draw = ImageDraw.Draw(rgb_image)
        radius = int(params[2*circles])
        centers = np.asarray(params[:2*circles]).reshape(-1, 2)
        colors = np.asarray(params[2*circles+1:]).reshape(-1, 3)
        for (x, y), color in zip(centers, colors):
            x, y = int(x), int(y)
            bounds = (x-radius, y-radius, x+radius, y+radius)
            draw.ellipse(bounds, fill=255)
            rgb_draw.ellipse(bounds, fill=tuple(map(int, color)))
        mask = np.asarray(mask_image) > 0
        rgb = np.asarray(rgb_image, dtype=np.float32)
        return -oracle.evaluate(mask, rgb)["score"]

    # Same wrapper and swarm size as author code, shortened only by the shared
    # external candidate/time budgets. Exceptions retain the best traced result.
    upstream.n_random_restart_pso(objective, low, high, swarmsize=swarm,
                                  maxiter=30, n_restarts=5)


def impact(oracle, upstream, seed: int, *, population: int = 12, phase1_fraction: float = 0.65):
    """Author DE mask/RGB kernels followed by fixed-mask pixel-level (1+1)-ES."""
    np.random.seed(seed)
    h, w = oracle.shape
    tile = oracle.grid_cell
    side = math.ceil(max(h, w) / tile)
    dimension = side * side
    tiles = max(3, min(dimension - 1, round(oracle.area_target * oracle.sign_pixels / tile**2)))
    clusters = min(3, tiles)
    population = max(4, min(population, max(4, oracle.candidate_limit // 8)))
    masks, colors = upstream.init_population(population, clusters, tiles, side * tile, tile)

    def decode(mask_code, color_code):
        mask = np.zeros((side * tile, side * tile), dtype=bool)
        rgb = np.zeros((side * tile, side * tile, 3), dtype=np.float32)
        for i, index in enumerate(np.flatnonzero(mask_code)):
            row, col = divmod(int(index), side)
            sl = np.s_[row*tile:(row+1)*tile, col*tile:(col+1)*tile]
            mask[sl] = True
            rgb[sl] = color_code[i]
        return mask[:h, :w], rgb[:h, :w]

    fitness = []
    for mask, color in zip(masks, colors):
        fitness.append(oracle.evaluate(*decode(mask, color))["score"])
    fitness = np.asarray(fitness)
    phase1_limit = max(population, int(oracle.candidate_limit * phase1_fraction))
    while oracle.evaluations + population <= phase1_limit:
        mutant = upstream.mutation(population, [masks, colors])
        # Author crossover samples an active mutant index. Guard its undefined
        # all-zero case, without changing any nonempty mutant or its fitness.
        for i in range(population):
            if not np.any(mutant[0][i]):
                mutant[0][i] = masks[i].copy()
        trials = upstream.crossover(population, mutant, [masks, colors], clusters, tiles, dimension)
        for i in range(population):
            score = oracle.evaluate(*decode(trials[0][i], trials[1][i]))["score"]
            if score > fitness[i]:
                masks[i], colors[i], fitness[i] = trials[0][i], trials[1][i], score
    index = int(np.argmax(fitness))
    mask, texture = decode(masks[index], colors[index])
    score = float(fitness[index])
    # The author's sigma=0.1 acts in ImageNet-normalized RGB space. Convert it
    # to byte RGB instead of mistakenly applying sigma=0.1/255 to pixels.
    sigma_rgb = 0.1 * 255 * np.asarray([0.229, 0.224, 0.225])
    while True:
        noise = np.random.normal(size=texture.shape).astype(np.float32) * sigma_rgb
        proposal = np.clip(texture + noise, 0, 255)
        result = oracle.evaluate(mask, proposal)
        if result["score"] > score:
            texture, score = proposal, result["score"]
