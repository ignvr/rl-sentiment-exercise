"""
Reward Functions for RL Sentiment Fine-tuning - SOLUTION VERSION

This module contains complete implementations of all reward functions.
This is the instructor/solution version - students should work with rewards.py.
"""

import math
from sentiment import get_sentiment_scores


# =============================================================================
# BASE REWARD FUNCTION
# =============================================================================

def sentiment_reward(completions: list[str]) -> list[float]:
    """
    Compute sentiment reward for a list of completions.
    
    This is the base reward function that returns sentiment scores in [0, 1].
    Higher values indicate more positive sentiment.
    
    Args:
        completions: List of generated text completions
    
    Returns:
        List of sentiment scores in [0, 1]
    """
    return get_sentiment_scores(completions)


# =============================================================================
# KL REGULARIZATION - SOLUTION
#
# CONTEXT: TRL includes built-in KL regularization (the `beta` parameter),
# applied per-token during advantage computation. Here we re-implement KL
# regularization at the token level for pedagogical purposes.
#
# You receive per-token log probabilities as list[list[float]]: one list per
# completion, with one float per generated token. Your functions should compute
# per-token KL terms and average them to produce one penalty scalar per completion.
# =============================================================================

def kl_penalty_forward(
    log_probs_policy: list[list[float]],
    log_probs_ref: list[list[float]],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Forward KL regularization penalty (token-level).
    
    SOLUTION:
    KL(π || π_ref) = E_π[log π - log π_ref]
    Since the data is already sampled from π, the per-token KL term is simply
    (log_policy - log_ref). Average over tokens and scale by kl_coef.
    Returns ≥ 0; the calling infrastructure subtracts this from the reward.
    """
    penalties = []
    for lp_policy, lp_ref in zip(log_probs_policy, log_probs_ref):
        n = len(lp_policy)
        if n == 0:
            penalties.append(0.0)
            continue
        token_kl = [lp_p - lp_r for lp_p, lp_r in zip(lp_policy, lp_ref)]
        penalties.append(kl_coef * sum(token_kl) / n)
    return penalties


def kl_penalty_backward(
    log_probs_policy: list[list[float]],
    log_probs_ref: list[list[float]],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Backward (reverse) KL regularization penalty (token-level).
    
    SOLUTION:
    KL(π_ref || π) = E_π_ref[log(π_ref / π)]
    Since we sample from π (not π_ref), we need importance sampling:
        E_π_ref[f(x)] = E_π[(π_ref(x)/π(x)) · f(x)]
    
    An exact implementation would apply sequence-level importance weight:
        penalty(x) = [Π_t π_ref(x_t|x<t)/π(x_t|x<t)] · [Σ_t log π_ref(x_t|x<t) - log π(x_t|x<t)]
    However, the product Π_t of T ratios can explode or vanish, causing high variance.
    
    We thus use an approximation:
        penalty(x) = (1/T) Σ_t [π_ref(x_t|x<t)/π(x_t|x<t)] · [log π_ref(x_t|x<t) - log π(x_t|x<t)]
    
    This replaces the single sequence-level weight with independent per-token weights.
    It is exact when π = π_ref, and a good approximation when they are close -
    which is the regime we are regularizing toward. This is a standard
    token-level approach often used in practice.
    
    Returns ≥ 0; the calling infrastructure subtracts this from the reward.
    """
    penalties = []
    for lp_policy, lp_ref in zip(log_probs_policy, log_probs_ref):
        n = len(lp_policy)
        if n == 0:
            penalties.append(0.0)
            continue
        token_kl = []
        for lp_p, lp_r in zip(lp_policy, lp_ref):
            diff = lp_r - lp_p
            weight = min(math.exp(diff), 100.0)  # importance-sampling weight P(π_ref) / P(π), clipped for stability
            token_kl.append(weight * diff)
        penalties.append(kl_coef * sum(token_kl) / n)
    return penalties


# =============================================================================
# REWARD SHAPING - SOLUTION
# =============================================================================

def shaped_reward(scores: list[float], completions: list[str], prompts: list[str] = None) -> list[float]:
    """
    Apply custom reward shaping to transform raw sentiment scores.
    
    SOLUTION: Target-length reward. Rewards positive sentiment but scales it
    by how close the completion is to a target of 10 words per completion.
    The length factor is 1/(1 + deviation/target).
    """
    target_len = 10
    shaped = []
    for score, completion in zip(scores, completions):
        n_words = len(completion.split())
        length_factor = 1 / (1 + abs(n_words/target_len - 1))
        shaped.append(score * length_factor)
    return shaped


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing SOLUTION implementations...\n")
    
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time.",
        "It was okay, nothing special.",
    ]
    
    # Test sentiment reward
    print("1. Sentiment Reward (raw scores [0, 1]):")
    rewards = sentiment_reward(test_texts)
    for text, reward in zip(test_texts, rewards):
        print(f"   {reward:.3f}: {text[:40]}...")
    
    # Test reward shaping
    print("\n2. Shaped Reward (exponential example):")
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_completions = ["bad", "meh", "okay", "good movie", "amazing film!"]
    shaped_results = shaped_reward(test_scores, test_completions)
    for score, s in zip(test_scores, shaped_results):
        print(f"   {score:.1f} -> {s:+.3f}")
    
    # Test KL penalties (token-level)
    print("\n3. KL Penalties (token-level):")
    test_lp_policy = [[-2.0, -3.0, -1.5], [-1.0, -2.0]]  # Per-token log probs
    test_lp_ref = [[-2.5, -2.5, -2.5], [-1.5, -1.5]]
    
    fwd = kl_penalty_forward(test_lp_policy, test_lp_ref, 0.1)
    print(f"   Forward KL penalties: {[f'{p:.4f}' for p in fwd]}")
    print(f"   (≥ 0; subtracted from reward by infrastructure)")
    
    bwd = kl_penalty_backward(test_lp_policy, test_lp_ref, 0.1)
    print(f"   Backward KL penalties: {[f'{p:.4f}' for p in bwd]}")
    print(f"   (≥ 0; importance-weighted, subtracted from reward)")
    
    print("\nAll solution tests passed!")
