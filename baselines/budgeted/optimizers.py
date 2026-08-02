"""Deterministic native optimizers using the shared budgeted evaluator."""
from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata
import math
import operator
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .protocol import BudgetedEvaluator, CandidateEvaluation, SearchResult, candidate_rank


def _evaluation_allowed(
    evaluator: BudgetedEvaluator, count: int, max_evaluations: Optional[int]
) -> bool:
    return evaluator.can_evaluate_more() and (
        max_evaluations is None or int(count) < int(max_evaluations)
    )


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if normalized < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    return int(normalized)


def _finite(name: str, value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _validate_common(seed: int, max_evaluations: Optional[int]) -> Tuple[int, Optional[int]]:
    normalized_seed = _integer("seed", seed, minimum=0)
    normalized_max = (
        None
        if max_evaluations is None
        else _integer("max_evaluations", max_evaluations, minimum=0)
    )
    return normalized_seed, normalized_max


def _finish(
    evaluator: BudgetedEvaluator,
    *,
    method: str,
    seed: int,
    implementation: str,
    config: Dict[str, Any],
    fidelity: str = "native_implementation",
    note: Optional[str] = None,
) -> SearchResult:
    status = "query_budget_exhausted" if not evaluator.can_evaluate_more() else "completed"
    return evaluator.result(
        method=method,
        seed=seed,
        implementation=implementation,
        config=config,
        fidelity=fidelity,
        status=status,
        note=note,
    )


def _repair_by_priority(
    priorities: Sequence[float], evaluator: BudgetedEvaluator
) -> Tuple[int, ...]:
    """Decode positive priorities under the exact, nonuniform pixel budget."""
    eligible = np.asarray(evaluator.selectable_indices, dtype=np.int64)
    values = np.asarray(priorities, dtype=np.float64)
    if values.shape != (eligible.size,):
        raise ValueError("priority vector has the wrong shape")
    # Stable secondary ordering by canonical index makes ties reproducible.
    order = sorted(
        range(eligible.size),
        key=lambda pos: (-float(values[pos]), int(eligible[pos])),
    )
    selected: List[int] = []
    used_groups: Set[int] = set()
    used = 0
    limit = int(evaluator.budget.material_pixel_limit)
    for pos in order:
        if float(values[pos]) <= 0.0:
            continue
        index = int(eligible[pos])
        group = evaluator.group_id(index)
        if group in used_groups:
            continue
        cost = int(evaluator.cell_material_pixels[index])
        if used + cost <= limit:
            selected.append(index)
            used_groups.add(group)
            used += cost
    return tuple(sorted(selected))


def _random_candidate(
    rng: np.random.Generator, evaluator: BudgetedEvaluator
) -> Tuple[int, ...]:
    """Sample physical groups, then one categorical token per chosen group."""

    n = len(evaluator.selectable_indices)
    inclusion_rate = float(rng.uniform(0.0, 1.0))
    priorities = np.full(n, -1.0, dtype=np.float64)
    positions_by_group: Dict[int, List[int]] = {}
    for position, index in enumerate(evaluator.selectable_indices):
        positions_by_group.setdefault(evaluator.group_id(index), []).append(position)
    for positions in positions_by_group.values():
        if float(rng.random()) >= inclusion_rate:
            continue
        selected_position = positions[int(rng.integers(0, len(positions)))]
        priorities[selected_position] = 1.0 + float(rng.random())
    return _repair_by_priority(priorities, evaluator)


def random_search(
    evaluator: BudgetedEvaluator,
    *,
    seed: int,
    max_evaluations: Optional[int] = None,
) -> SearchResult:
    """Independent group masks with uniform categorical material choices."""
    seed, max_evaluations = _validate_common(seed, max_evaluations)
    rng = np.random.default_rng(seed)
    count = 0
    while _evaluation_allowed(evaluator, count, max_evaluations):
        evaluator.evaluate(_random_candidate(rng, evaluator))
        count += 1
    return _finish(
        evaluator,
        method="random_search",
        seed=seed,
        implementation="native-grouped-categorical-random-v1",
        config={"max_evaluations": max_evaluations},
    )


def greedy_search(
    evaluator: BudgetedEvaluator,
    *,
    seed: int,
    max_evaluations: Optional[int] = None,
    evaluate_empty: bool = True,
) -> SearchResult:
    """Forward greedy addition with every marginal scored by the same oracle."""
    seed, max_evaluations = _validate_common(seed, max_evaluations)
    if type(evaluate_empty) is not bool:
        raise ValueError("evaluate_empty must be a boolean")
    rng = np.random.default_rng(seed)
    count = 0
    current: Tuple[int, ...] = ()
    if evaluate_empty and _evaluation_allowed(evaluator, count, max_evaluations):
        evaluator.evaluate(current)
        count += 1

    while _evaluation_allowed(evaluator, count, max_evaluations):
        current_set = set(current)
        current_groups = {evaluator.group_id(index) for index in current}
        additions = [
            index
            for index in evaluator.selectable_indices
            if index not in current_set
            and evaluator.group_id(index) not in current_groups
            and evaluator.is_material_feasible(current + (index,))
        ]
        if not additions:
            break
        # Seeded ordering prevents a fixed low-index advantage when the final
        # marginal sweep is cut short by the query limit.
        additions = [int(value) for value in rng.permutation(additions)]
        sweep: List[CandidateEvaluation] = []
        for index in additions:
            if not _evaluation_allowed(evaluator, count, max_evaluations):
                break
            row = evaluator.evaluate(current + (index,))
            sweep.append(row)
            count += 1
        if not sweep:
            break
        current = max(sweep, key=candidate_rank).selected_indices

    return _finish(
        evaluator,
        method="forward_greedy",
        seed=seed,
        implementation="native-forward-addition-v1",
        config={
            "max_evaluations": max_evaluations,
            "evaluate_empty": bool(evaluate_empty),
        },
    )


def _mutate_genome(
    genome: Set[int],
    rng: np.random.Generator,
    evaluator: BudgetedEvaluator,
    mutation_rate: float,
) -> Tuple[int, ...]:
    priorities = np.full(len(evaluator.selectable_indices), -1.0, dtype=np.float64)
    for position, index in enumerate(evaluator.selectable_indices):
        active = index in genome
        if float(rng.random()) < float(mutation_rate):
            active = not active
        priorities[position] = float(rng.random()) + (1.0 if active else -1.0)
    return _repair_by_priority(priorities, evaluator)


def genetic_search(
    evaluator: BudgetedEvaluator,
    *,
    seed: int,
    population_size: int = 16,
    elite_fraction: float = 0.25,
    mutation_rate: Optional[float] = None,
    max_evaluations: Optional[int] = None,
) -> SearchResult:
    """Binary genetic algorithm with uniform crossover and exact repair."""
    seed, max_evaluations = _validate_common(seed, max_evaluations)
    population_size = _integer("population_size", population_size, minimum=2)
    elite_fraction = _finite("elite_fraction", elite_fraction)
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    dimension = len(evaluator.selectable_indices)
    rate = _finite("mutation_rate", mutation_rate) if mutation_rate is not None else 1.0 / dimension
    if not 0.0 <= rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1]")

    population = [
        _random_candidate(rng, evaluator) for _ in range(int(population_size))
    ]
    count = 0
    generations = 0
    while _evaluation_allowed(evaluator, count, max_evaluations):
        evaluated: List[CandidateEvaluation] = []
        for genome in population:
            if not _evaluation_allowed(evaluator, count, max_evaluations):
                break
            evaluated.append(evaluator.evaluate(genome))
            count += 1
        if len(evaluated) < 2:
            break
        generations += 1
        ranked = sorted(evaluated, key=candidate_rank, reverse=True)
        elite_count = max(1, int(np.ceil(len(ranked) * float(elite_fraction))))
        elites = [set(row.selected_indices) for row in ranked[:elite_count]]
        parents = [set(row.selected_indices) for row in ranked[: max(2, len(ranked) // 2)]]
        next_population: List[Tuple[int, ...]] = [
            tuple(sorted(genome)) for genome in elites
        ]
        while len(next_population) < int(population_size):
            first = parents[int(rng.integers(0, len(parents)))]
            second = parents[int(rng.integers(0, len(parents)))]
            union = sorted(first | second)
            child = {
                index
                for index in union
                if (index in first and index in second) or bool(rng.integers(0, 2))
            }
            next_population.append(
                _mutate_genome(child, rng, evaluator, rate)
            )
        population = next_population[: int(population_size)]

    return _finish(
        evaluator,
        method="genetic_algorithm",
        seed=seed,
        implementation="native-binary-ga-v1",
        config={
            "population_size": int(population_size),
            "elite_fraction": float(elite_fraction),
            "mutation_rate": rate,
            "generations_completed": generations,
            "max_evaluations": max_evaluations,
        },
    )


def evolution_strategy_search(
    evaluator: BudgetedEvaluator,
    *,
    seed: int,
    population_size: int = 16,
    elite_fraction: float = 0.25,
    sigma: float = 1.0,
    learning_rate: float = 0.35,
    sigma_decay: float = 0.98,
    max_evaluations: Optional[int] = None,
) -> SearchResult:
    """Native Gaussian evolution strategy over binary-mask logits.

    This is reported as ``gaussian_es`` and is not mislabeled as CMA-ES.
    """
    seed, max_evaluations = _validate_common(seed, max_evaluations)
    population_size = _integer("population_size", population_size, minimum=2)
    elite_fraction = _finite("elite_fraction", elite_fraction)
    sigma = _finite("sigma", sigma)
    learning_rate = _finite("learning_rate", learning_rate)
    sigma_decay = _finite("sigma_decay", sigma_decay)
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1]")
    if sigma <= 0.0 or learning_rate <= 0.0:
        raise ValueError("sigma and learning_rate must be positive")
    if not 0.0 < sigma_decay <= 1.0:
        raise ValueError("sigma_decay must be in (0, 1]")

    rng = np.random.default_rng(seed)
    n = len(evaluator.selectable_indices)
    mean = np.full(n, -0.5, dtype=np.float64)
    current_sigma = float(sigma)
    count = 0
    generations = 0
    while _evaluation_allowed(evaluator, count, max_evaluations):
        noises = rng.normal(size=(int(population_size), n))
        evaluated: List[Tuple[CandidateEvaluation, np.ndarray]] = []
        for noise in noises:
            if not _evaluation_allowed(evaluator, count, max_evaluations):
                break
            candidate = _repair_by_priority(
                mean + current_sigma * noise, evaluator
            )
            evaluated.append((evaluator.evaluate(candidate), noise))
            count += 1
        if len(evaluated) < 2:
            break
        generations += 1
        evaluated.sort(key=lambda pair: candidate_rank(pair[0]), reverse=True)
        elite_count = max(1, int(np.ceil(len(evaluated) * float(elite_fraction))))
        elite_noise = np.stack([pair[1] for pair in evaluated[:elite_count]])
        mean += float(learning_rate) * current_sigma * elite_noise.mean(axis=0)
        mean = np.clip(mean, -8.0, 8.0)
        current_sigma = max(0.05, current_sigma * float(sigma_decay))

    return _finish(
        evaluator,
        method="gaussian_es",
        seed=seed,
        implementation="native-isotropic-logit-es-v1",
        config={
            "population_size": int(population_size),
            "elite_fraction": float(elite_fraction),
            "initial_sigma": float(sigma),
            "final_sigma": float(current_sigma),
            "learning_rate": float(learning_rate),
            "sigma_decay": float(sigma_decay),
            "generations_completed": generations,
            "max_evaluations": max_evaluations,
        },
    )


def fipatch_style_pso_proxy_search(
    evaluator: BudgetedEvaluator,
    *,
    seed: int,
    swarm_size: int = 16,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
    max_evaluations: Optional[int] = None,
) -> SearchResult:
    """Binary/discrete PSO family proxy under exact shared budgets.

    The implementation is deliberately named and tagged as a proxy.  It is
    useful for an executable PSO-family comparison but is **not** presented as
    a reproduction of FIPatch's full optimizer or physical model.
    """
    seed, max_evaluations = _validate_common(seed, max_evaluations)
    swarm_size = _integer("swarm_size", swarm_size, minimum=2)
    inertia = _finite("inertia", inertia)
    cognitive = _finite("cognitive", cognitive)
    social = _finite("social", social)
    if swarm_size < 2:
        raise ValueError("swarm_size must be at least 2")
    for name, value in (
        ("inertia", inertia),
        ("cognitive", cognitive),
        ("social", social),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    rng = np.random.default_rng(seed)
    n = len(evaluator.selectable_indices)
    positions = rng.random((int(swarm_size), n)) < 0.2
    velocities = rng.normal(0.0, 0.25, size=(int(swarm_size), n))
    personal_positions = positions.copy()
    personal_rows: List[Optional[CandidateEvaluation]] = [
        None for _ in range(int(swarm_size))
    ]
    global_position = positions[0].copy()
    global_row: Optional[CandidateEvaluation] = None
    count = 0
    iterations = 0

    while _evaluation_allowed(evaluator, count, max_evaluations):
        complete_iteration = True
        for particle in range(int(swarm_size)):
            if not _evaluation_allowed(evaluator, count, max_evaluations):
                complete_iteration = False
                break
            priorities = np.where(
                positions[particle],
                1.0 + rng.random(n),
                -1.0 + rng.random(n),
            )
            candidate = _repair_by_priority(priorities, evaluator)
            # Synchronize the particle with exact repair before its velocity
            # update; otherwise infeasible bits would exert hidden influence.
            selected = set(candidate)
            positions[particle] = np.asarray(
                [
                    index in selected
                    for index in evaluator.selectable_indices
                ],
                dtype=bool,
            )
            row = evaluator.evaluate(candidate)
            count += 1
            previous = personal_rows[particle]
            if previous is None or candidate_rank(row) > candidate_rank(previous):
                personal_rows[particle] = row
                personal_positions[particle] = positions[particle].copy()
            if global_row is None or candidate_rank(row) > candidate_rank(global_row):
                global_row = row
                global_position = positions[particle].copy()
        if not complete_iteration:
            break
        iterations += 1
        r1 = rng.random((int(swarm_size), n))
        r2 = rng.random((int(swarm_size), n))
        numeric_positions = positions.astype(np.float64)
        velocities = (
            float(inertia) * velocities
            + float(cognitive)
            * r1
            * (personal_positions.astype(np.float64) - numeric_positions)
            + float(social)
            * r2
            * (global_position.astype(np.float64)[None, :] - numeric_positions)
        )
        velocities = np.clip(velocities, -12.0, 12.0)
        probabilities = 1.0 / (1.0 + np.exp(-velocities))
        positions = rng.random((int(swarm_size), n)) < probabilities

    return _finish(
        evaluator,
        method="fipatch_style_pso_proxy",
        seed=seed,
        implementation="native-binary-pso-family-proxy-v1",
        fidelity="proxy_not_fipatch_reproduction",
        config={
            "swarm_size": int(swarm_size),
            "inertia": float(inertia),
            "cognitive": float(cognitive),
            "social": float(social),
            "iterations_completed": iterations,
            "max_evaluations": max_evaluations,
        },
        note=(
            "Executable binary PSO family proxy only. A named FIPatch result "
            "requires a pinned runner, declared candidate mapping, and "
            "independent semantic audit."
        ),
    )


def cma_es_search(
    evaluator: BudgetedEvaluator,
    *,
    seed: int,
    population_size: int = 16,
    sigma: float = 1.0,
    max_evaluations: Optional[int] = None,
) -> SearchResult:
    """Use the optional reference ``cma`` package, or report unavailable.

    There is intentionally no hand-written algorithm called CMA-ES.  If the
    reference package is absent, the result is a machine-readable unavailable
    row and consumes no candidate queries.
    """
    seed, max_evaluations = _validate_common(seed, max_evaluations)
    population_size = _integer("population_size", population_size, minimum=2)
    sigma = _finite("sigma", sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    try:
        cma = importlib.import_module("cma")
    except ImportError:
        return evaluator.result(
            method="cma_es",
            seed=seed,
            implementation="optional-cma-package",
            fidelity="reference_package",
            config={
                "population_size": int(population_size),
                "sigma": float(sigma),
                "max_evaluations": max_evaluations,
            },
            status="unavailable",
            note=(
                "Install the declared 'cma' dependency to run its reference "
                "CMA-ES implementation over the common repaired-priority space."
            ),
        )
    n = len(evaluator.selectable_indices)
    strategy = cma.CMAEvolutionStrategy(
        [-0.5] * n,
        float(sigma),
        {
            "seed": int(seed),
            "popsize": int(population_size),
            "verbose": -9,
        },
    )
    count = 0
    generations = 0
    while _evaluation_allowed(evaluator, count, max_evaluations):
        points = strategy.ask()
        rows: List[CandidateEvaluation] = []
        used_points: List[Sequence[float]] = []
        for point in points:
            if not _evaluation_allowed(evaluator, count, max_evaluations):
                break
            row = evaluator.evaluate(_repair_by_priority(point, evaluator))
            rows.append(row)
            used_points.append(point)
            count += 1
        # CMA's update requires the whole requested population.  A partial
        # terminal generation is evaluated and logged but never passed to tell.
        if len(rows) != len(points):
            break
        strategy.tell(used_points, [-float(row.score) for row in rows])
        generations += 1

    try:
        package_version = importlib_metadata.version("cma")
    except importlib_metadata.PackageNotFoundError:
        package_version = "unknown"
    return _finish(
        evaluator,
        method="cma_es",
        seed=seed,
        implementation=f"cma-package-{package_version}",
        fidelity="reference_package",
        config={
            "population_size": int(population_size),
            "sigma": float(sigma),
            "generations_completed": generations,
            "max_evaluations": max_evaluations,
        },
    )
