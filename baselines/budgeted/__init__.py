"""Query- and material-budget matched black-box optimizers.

The public API is deliberately independent of Gym and detector packages.  A
candidate oracle is the only component allowed to query a detector; every
optimizer therefore receives the same observations and passes through the
same budget guard.
"""

from .protocol import (
    BudgetExhausted,
    BudgetSpec,
    BudgetedEvaluator,
    CandidateEvaluation,
    CandidateOracle,
    MaterialBudgetExceeded,
    OracleObservation,
    SearchResult,
)
from .optimizers import (
    cma_es_search,
    evolution_strategy_search,
    fipatch_style_pso_proxy_search,
    genetic_search,
    greedy_search,
    random_search,
)

__all__ = [
    "BudgetExhausted",
    "BudgetSpec",
    "BudgetedEvaluator",
    "CandidateEvaluation",
    "CandidateOracle",
    "MaterialBudgetExceeded",
    "OracleObservation",
    "SearchResult",
    "cma_es_search",
    "evolution_strategy_search",
    "fipatch_style_pso_proxy_search",
    "genetic_search",
    "greedy_search",
    "random_search",
]
