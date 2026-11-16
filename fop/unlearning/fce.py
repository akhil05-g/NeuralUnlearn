"""
phase5_fce_module.py - Phase 5: Fuzzy Channel Erasure (FCE) Engine.

THE CORE RESEARCH CONTRIBUTION:
Inverts the channel-wise knowledge preservation gating used in incremental small-object detection
(IncSOD-Net). Channels salient for the target class (windmill) but weakly used by retained classes
are selectively suppressed, while shared contextual representations are protected.

Mathematical formulation:
1. Fuzzy Membership Functions: Low, Medium, High for I_forget and I_retain.
2. Mamdani Fuzzy Rules:
   - R1: IF I_forget is HIGH and I_retain is LOW    -> ERASE = STRONG  (0.95)
   - R2: IF I_forget is HIGH and I_retain is MEDIUM -> ERASE = MODERATE(0.65)
   - R3: IF I_forget is HIGH and I_retain is HIGH   -> ERASE = PROTECT (0.05) [Shared channel]
   - R4: IF I_forget is MED  and I_retain is LOW    -> ERASE = MODERATE(0.50)
   - R5: IF I_forget is MED  and I_retain is MEDIUM -> ERASE = MILD    (0.25)
   - R6: IF I_forget is MED  and I_retain is HIGH   -> ERASE = PROTECT (0.05)
   - R7: IF I_forget is LOW                         -> ERASE = NONE    (0.00)
3. Defuzzification via firing-strength weighted average.
4. Channel Scaling: M(c) = 1.0 - lambda * E(c).
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    HAS_TORCH = False

from fop.config import (
    FEATURES_DIR,
    PLOTS_DIR,
    DEFAULT_ERASURE_STRENGTH,
    ERASURE_STRENGTHS,
    DEVICE,
)
from fop.utils import setup_logging, ensure_dirs, save_results

logger = setup_logging("phase5_fce.log")


class FuzzyChannelErasure:
    """
    Fuzzy Channel Erasure (FCE) engine that calculates channel-wise suppression weights.
    """
    def __init__(
        self,
        forget_importance: np.ndarray,
        retain_importance: np.ndarray,
        erasure_strength: float = DEFAULT_ERASURE_STRENGTH,
    ):
        """
        Args:
            forget_importance: 1D numpy array of normalized [0, 1] importance scores for forget class.
            retain_importance: 1D numpy array of normalized [0, 1] importance scores for retained classes.
            erasure_strength: Scaling factor lambda in [0.0, 1.0] controlling unlearning aggressiveness.
        """
        assert len(forget_importance) == len(retain_importance), "Importance vector dimensions must match."
        self.num_channels = len(forget_importance)
        self.forget_imp = np.clip(forget_importance, 0.0, 1.0)
        self.retain_imp = np.clip(retain_importance, 0.0, 1.0)
        self.erasure_strength = float(erasure_strength)

        # Predefined rule consequence weights
        self.consequents = {
            "strong": 0.95,
            "moderate": 0.60,
            "mild": 0.30,
            "protect": 0.05,
            "none": 0.00,
        }

        self.erasure_weights = self._compute_channel_erasure_weights()

    @staticmethod
    def _membership_low(x: np.ndarray) -> np.ndarray:
        """Trapezoidal/descending membership for LOW importance [0.0, 0.45]."""
        return np.clip((0.45 - x) / (0.45 - 0.15 + 1e-6), 0.0, 1.0)

    @staticmethod
    def _membership_medium(x: np.ndarray) -> np.ndarray:
        """Triangular membership for MEDIUM importance centered at 0.50."""
        left = (x - 0.20) / (0.50 - 0.20 + 1e-6)
        right = (0.80 - x) / (0.80 - 0.50 + 1e-6)
        return np.clip(np.minimum(left, right), 0.0, 1.0)

    @staticmethod
    def _membership_high(x: np.ndarray) -> np.ndarray:
        """Trapezoidal/ascending membership for HIGH importance [0.45, 1.0]."""
        return np.clip((x - 0.45) / (0.75 - 0.45 + 1e-6), 0.0, 1.0)

    def _compute_channel_erasure_weights(self) -> np.ndarray:
        """
        Apply Mamdani fuzzy inference rules and defuzzify to bounded erasure weights [0, 1].
        """
        f_low = self._membership_low(self.forget_imp)
        f_med = self._membership_medium(self.forget_imp)
        f_high = self._membership_high(self.forget_imp)

        r_low = self._membership_low(self.retain_imp)
        r_med = self._membership_medium(self.retain_imp)
        r_high = self._membership_high(self.retain_imp)

        # Rule evaluation (Mamdani min operator for conjunctions)
        # R1: High forget AND Low retain -> Strong
        alpha_1 = np.minimum(f_high, r_low)
        # R2: High forget AND Med retain -> Moderate
        alpha_2 = np.minimum(f_high, r_med)
        # R3: High forget AND High retain -> Protect (Shared channel)
        alpha_3 = np.minimum(f_high, r_high)
        # R4: Med forget AND Low retain -> Moderate
        alpha_4 = np.minimum(f_med, r_low)
        # R5: Med forget AND Med retain -> Mild
        alpha_5 = np.minimum(f_med, r_med)
        # R6: Med forget AND High retain -> Protect
        alpha_6 = np.minimum(f_med, r_high)
        # R7: Low forget -> None
        alpha_7 = f_low

        numerator = (
            alpha_1 * self.consequents["strong"]
            + alpha_2 * self.consequents["moderate"]
            + alpha_3 * self.consequents["protect"]
            + alpha_4 * self.consequents["moderate"]
            + alpha_5 * self.consequents["mild"]
            + alpha_6 * self.consequents["protect"]
            + alpha_7 * self.consequents["none"]
        )
        denominator = alpha_1 + alpha_2 + alpha_3 + alpha_4 + alpha_5 + alpha_6 + alpha_7 + 1e-6

        raw_erasure = numerator / denominator
        # Scale by global erasure strength lambda
        scaled_erasure = np.clip(raw_erasure * self.erasure_strength, 0.0, 1.0)
        return scaled_erasure

    def get_erasure_weights(self) -> np.ndarray:
        """Return 1D array of per-channel erasure weights E(c) in [0, 1]."""
        return self.erasure_weights

    def get_channel_multipliers(self) -> np.ndarray:
        """Return 1D array of channel scaling multipliers M(c) = 1.0 - E(c) in [0, 1]."""
        return 1.0 - self.erasure_weights

    def apply_to_tensor(self, feature_tensor: Any) -> Any:
        """
        Apply channel erasure to a 4D feature tensor [B, C, H, W].
        Multiplies channel c by (1.0 - E(c)).
        """
        if torch is None:
            return feature_tensor
        multipliers = torch.from_numpy(self.get_channel_multipliers()).to(
            dtype=feature_tensor.dtype, device=feature_tensor.device
        )
        # Shape: [1, C, 1, 1]
        scale = multipliers.view(1, -1, 1, 1)
        return feature_tensor * scale


class FCEForwardHook:
    """
    PyTorch forward hook that dynamically scales activations during forward pass.
    """
    def __init__(self, fce_module: FuzzyChannelErasure):
        self.fce = fce_module
        self.enabled = True
        self.hook_handle = None

    def __call__(self, module, input_tensor, output_tensor):
        if not self.enabled:
            return output_tensor

        if isinstance(output_tensor, tuple):
            first = self.fce.apply_to_tensor(output_tensor[0])
            return (first,) + output_tensor[1:]
        elif isinstance(output_tensor, list):
            output_tensor[0] = self.fce.apply_to_tensor(output_tensor[0])
            return output_tensor
        else:
            return self.fce.apply_to_tensor(output_tensor)

    def register(self, module: Any):
        if hasattr(module, "register_forward_hook"):
            self.hook_handle = module.register_forward_hook(self)
            logger.info("FCE forward hook attached to %s", module.__class__.__name__)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


def apply_fce_to_model_weights(
    yolo_model,
    fce_module: FuzzyChannelErasure,
    layer_indices: List[int] = [12, 15, 18],
) -> None:
    """
    Directly scale the convolution weights of selected neck layers by the FCE channel multipliers.
    This creates an erased checkpoint that does not require runtime hooks.
    """
    multipliers = torch.from_numpy(fce_module.get_channel_multipliers()).float()

    for idx in layer_indices:
        if idx >= len(yolo_model.model.model):
            continue
        layer = yolo_model.model.model[idx]
        # Search for Conv2d layers inside the block
        for name, module in layer.named_modules():
            if isinstance(module, nn.Conv2d):
                out_channels = module.weight.shape[0]
                if out_channels == len(multipliers):
                    scale = multipliers.to(module.weight.device).view(-1, 1, 1, 1)
                    with torch.no_grad():
                        module.weight.data.mul_(scale)
                        if module.bias is not None:
                            module.bias.data.mul_(multipliers.to(module.bias.device))
                    logger.info("Applied FCE channel scaling directly to layer %d (%s)", idx, name)
                    break


def visualize_fuzzy_rules_and_weights(
    fce: FuzzyChannelErasure,
    save_dir: Path = PLOTS_DIR / "feature_analysis",
) -> None:
    """Plot membership functions and calculated channel erasure weights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot Membership Functions
    x = np.linspace(0.0, 1.0, 200)
    plt.figure(figsize=(9, 4.5))
    plt.plot(x, fce._membership_low(x), label="Low", color="#2A9D8F", lw=2.5)
    plt.plot(x, fce._membership_medium(x), label="Medium", color="#E76F51", lw=2.5)
    plt.plot(x, fce._membership_high(x), label="High", color="#E63946", lw=2.5)
    plt.title("Fuzzy Membership Functions for Channel Importance (I_forget & I_retain)")
    plt.xlabel("Normalized Importance Score")
    plt.ylabel("Degree of Membership")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "fuzzy_membership_functions.png", dpi=300)
    plt.close()

    # 2. Plot Computed Erasure Weights per Channel
    weights = fce.get_erasure_weights()
    multipliers = fce.get_channel_multipliers()
    chans = np.arange(len(weights))

    plt.figure(figsize=(13, 5))
    colors = ["#E63946" if w > 0.4 else "#2A9D8F" if w < 0.1 else "#F4A261" for w in weights]
    plt.bar(chans, weights, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
    plt.axhline(0.4, color="red", linestyle="--", label="High Erasure Threshold (>0.4)")
    plt.axhline(0.1, color="green", linestyle="--", label="Protected Threshold (<0.1)")
    plt.xlabel("Channel Index")
    plt.ylabel("Erasure Weight E(c)")
    plt.title(f"FCE Channel Erasure Spectrum (Strength lambda={fce.erasure_strength})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "fce_channel_erasure_weights.png", dpi=300)
    plt.close()

    logger.info("Saved FCE diagnostic plots to %s", save_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Fuzzy Channel Erasure (FCE) Engine.")
    parser.add_argument("--strength", type=float, default=DEFAULT_ERASURE_STRENGTH,
                        help="Global erasure strength parameter lambda")
    args = parser.parse_args()

    npy_path = FEATURES_DIR / "channel_importance.npy"
    if npy_path.exists():
        data = np.load(npy_path, allow_pickle=True).item()
        forget_imp = data["forget_norm"]
        retain_imp = data["retain_norm"]
        logger.info("Loaded cached channel importance from %s", npy_path)
    else:
        logger.warning("Cached importance not found. Using synthetic channel scores.")
        num_c = 128
        forget_imp = np.random.uniform(0.1, 0.4, size=num_c)
        retain_imp = np.random.uniform(0.3, 0.9, size=num_c)
        # Force specific channels
        forget_imp[:12] = np.random.uniform(0.75, 0.95, size=12)
        retain_imp[:12] = np.random.uniform(0.05, 0.25, size=12)

    fce = FuzzyChannelErasure(forget_imp, retain_imp, erasure_strength=args.strength)
    weights = fce.get_erasure_weights()

    erased_count = np.sum(weights > 0.4)
    protected_count = np.sum(weights < 0.1)

    visualize_fuzzy_rules_and_weights(fce)

    print("\n" + "=" * 65)
    print(" PHASE 5: FUZZY CHANNEL ERASURE (FCE) MODULE")
    print("=" * 65)
    print(f"Total Profiled Channels      : {fce.num_channels}")
    print(f"Erasure Strength (lambda)    : {fce.erasure_strength}")
    print(f"Strongly Erased Channels (>0.4): {erased_count}")
    print(f"Protected Channels (<0.1)    : {protected_count}")
    print(f"Average Erasure Weight       : {np.mean(weights):.4f}")
    print(f"Plots saved to               : {PLOTS_DIR / 'feature_analysis'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
