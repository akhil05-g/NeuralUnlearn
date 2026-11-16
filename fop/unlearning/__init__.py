from fop.unlearning.fce import (
    FuzzyChannelErasure,
    FCEForwardHook,
    apply_fce_to_model_weights,
    visualize_fuzzy_rules_and_weights,
)
from fop.unlearning.trainer import run_fce_unlearning, run_erasure_sweep

__all__ = [
    "FuzzyChannelErasure",
    "FCEForwardHook",
    "apply_fce_to_model_weights",
    "visualize_fuzzy_rules_and_weights",
    "run_fce_unlearning",
    "run_erasure_sweep",
]
