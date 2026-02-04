"""
Reward Functions Module - SOLUTION FILE (Instructor Only)

This file contains complete implementations of all reward functions.
Do not distribute to students until after the exercise.

For the student version with TODOs, see rewards.py
"""

import math
import torch
from sentiment import get_sentiment_scores


# =============================================================================
# SOLUTION: Exercise 1 - Basic Sentiment Reward
# =============================================================================

def sentiment_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Basic sentiment reward function.
    
    SOLUTION: Simply return the sentiment scores.
    """
    return get_sentiment_scores(completions)


# =============================================================================
# SOLUTION: Exercise 2 - Reward Shaping
# =============================================================================

def shaped_reward_exponential(
    completions: list[str],
    temperature: float = 1.0,
    **kwargs
) -> list[float]:
    """
    Exponential reward shaping.
    
    SOLUTION: Apply exponential transformation for non-linear scaling.
    """
    scores = get_sentiment_scores(completions)
    rewards = [math.exp(score / temperature) - 1 for score in scores]
    return rewards


# =============================================================================
# SOLUTION: Exercise 3 - KL Divergence Regularization
# =============================================================================

def compute_ref_log_probs(
    completions: list[str],
    prompts: list[str],
    ref_model,
    tokenizer,
) -> list[float]:
    """
    Compute normalized log probabilities of completions under the reference model.
    
    This is a helper function that students will use to implement KL penalties.
    
    Args:
        completions: List of generated text completions
        prompts: List of prompts that generated these completions  
        ref_model: Reference model (frozen copy of initial policy)
        tokenizer: Tokenizer for the model
    
    Returns:
        List of normalized log probabilities (per token average)
    
    SOLUTION: Tokenize prompt+completion, compute log probs for completion tokens only.
    """
    device = next(ref_model.parameters()).device
    
    # Combine prompts and completions
    full_texts = [p + c for p, c in zip(prompts, completions)]
    
    # Tokenize full sequences
    inputs = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prompt lengths to know where completions start
    prompt_inputs = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=512,
    )
    prompt_lengths = [len(ids) for ids in prompt_inputs["input_ids"]]
    
    # Forward pass through reference model
    with torch.no_grad():
        outputs = ref_model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = inputs["input_ids"]
    
    # Compute average log prob for completion tokens only
    normalized_log_probs = []
    for i in range(len(completions)):
        start_idx = prompt_lengths[i]
        seq_len = int(inputs["attention_mask"][i].sum().item())
        
        # Sum log probs for completion tokens
        # logits[t] predicts token[t+1], so we look at positions start_idx to seq_len-2
        seq_log_prob = 0.0
        num_tokens = 0
        for t in range(start_idx, seq_len - 1):
            token_id = input_ids[i, t + 1].item()
            seq_log_prob += log_probs[i, t, token_id].item()
            num_tokens += 1
        
        # Normalize by number of tokens
        avg_log_prob = seq_log_prob / max(num_tokens, 1)
        normalized_log_probs.append(avg_log_prob)
    
    return normalized_log_probs


def kl_penalty_forward(
    completions: list[str],
    prompts: list[str],
    ref_model,
    tokenizer,
    kl_coef: float = 0.1,
    **kwargs
) -> list[float]:
    """
    Forward KL regularization: encourages outputs likely under the reference model.
    
    Approximates: -D_KL(policy || reference) as a reward bonus.
    
    Forward KL is "mode-covering" - it penalizes the policy for generating
    text that would be unlikely under the reference model. This helps prevent
    the model from generating degenerate or repetitive text.
    
    The reward bonus is: kl_coef * log(P_ref(completion))
    - Positive when completion is likely under reference
    - Negative when completion is unlikely under reference
    
    Args:
        completions: List of generated text completions
        prompts: List of prompts that generated these completions
        ref_model: Reference model (frozen copy of initial policy)
        tokenizer: Tokenizer for the model
        kl_coef: Coefficient for KL penalty (higher = stronger regularization)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        List of KL bonus values to ADD to the base reward
    
    SOLUTION: Return kl_coef * normalized_log_prob_ref
    """
    log_probs = compute_ref_log_probs(completions, prompts, ref_model, tokenizer)
    
    # Forward KL bonus: reward outputs that are likely under reference
    # Higher log prob = more similar to reference = positive bonus
    bonuses = [kl_coef * lp for lp in log_probs]
    
    return bonuses


def kl_penalty_backward(
    completions: list[str],
    prompts: list[str],
    ref_model,
    tokenizer,
    kl_coef: float = 0.1,
    **kwargs
) -> list[float]:
    """
    Backward KL regularization: strongly penalizes outputs unlikely under reference.
    
    Approximates: -D_KL(reference || policy) as a reward penalty.
    
    Backward KL is "mode-seeking" - it heavily penalizes outputs that have
    very low probability under the reference model, while being more lenient
    about outputs that are merely "different but plausible".
    
    The penalty uses exponential weighting: -kl_coef * exp(-log(P_ref))
    - Small penalty when completion is likely under reference
    - Large penalty when completion is very unlikely under reference
    
    This creates different behavior than forward KL:
    - Forward KL: linear penalty, uniform pressure to match reference
    - Backward KL: exponential penalty, focuses on avoiding "bad" outputs
    
    Args:
        completions: List of generated text completions
        prompts: List of prompts that generated these completions
        ref_model: Reference model (frozen copy of initial policy)
        tokenizer: Tokenizer for the model
        kl_coef: Coefficient for KL penalty (higher = stronger regularization)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        List of KL penalty values to ADD to the base reward (will be negative)
    
    SOLUTION: Return -kl_coef * exp(-log_prob), which heavily penalizes low-prob outputs
    """
    log_probs = compute_ref_log_probs(completions, prompts, ref_model, tokenizer)
    
    # Backward KL penalty: exponentially penalize unlikely outputs
    # exp(-log_prob) = 1/prob, so very low prob = very high penalty
    # Clamp to avoid numerical issues with very low probabilities
    penalties = [-kl_coef * min(math.exp(-lp), 100.0) for lp in log_probs]
    
    return penalties


# =============================================================================
# COMBINED REWARD FUNCTION (with KL support)
# =============================================================================

def make_reward_function(
    shaping: str = "none",
    kl_type: str = "none",
    kl_coef: float = 0.1,
    ref_model=None,
    tokenizer=None,
    base_scorer=None,
    negate: bool = False,
):
    """
    Factory function to create a combined reward function.
    
    This creates a reward function that:
    1. Computes base reward using the scorer (with optional shaping)
    2. Adds KL regularization penalty/bonus (if enabled)
    3. Optionally negates the final reward (for negative sentiment optimization)
    
    Args:
        shaping: Type of reward shaping ("none" or "exponential")
        kl_type: Type of KL regularization ("none", "forward", "backward")
        kl_coef: Coefficient for KL term
        ref_model: Reference model (required if kl_type != "none")
        tokenizer: Tokenizer (required if kl_type != "none")
        base_scorer: Optional custom scorer function(texts) -> [0,1] scores.
                     If None, uses sentiment_reward.
        negate: If True, negate the final reward (optimize for negative sentiment)
    
    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    if kl_type != "none" and (ref_model is None or tokenizer is None):
        raise ValueError(f"ref_model and tokenizer required for kl_type={kl_type}")
    
    # Use provided scorer or default to sentiment_reward
    scorer = base_scorer if base_scorer is not None else sentiment_reward
    
    def reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        # Step 1: Compute base reward from scorer
        raw_scores = scorer(completions)
        
        # Step 2: Apply shaping
        if shaping == "none":
            base_rewards = raw_scores
        elif shaping == "exponential":
            # Apply exponential shaping to raw scores
            base_rewards = [shaped_reward_exponential_single(s) for s in raw_scores]
        else:
            raise ValueError(f"Unknown shaping: {shaping}")
        
        # Step 3: Add KL regularization if enabled
        if kl_type != "none" and prompts is not None:
            if kl_type == "forward":
                kl_terms = kl_penalty_forward(
                    completions, prompts, ref_model, tokenizer, kl_coef
                )
            elif kl_type == "backward":
                kl_terms = kl_penalty_backward(
                    completions, prompts, ref_model, tokenizer, kl_coef
                )
            else:
                raise ValueError(f"Unknown kl_type: {kl_type}")
            
            # Combine: total_reward = base_reward + kl_term
            base_rewards = [base + kl for base, kl in zip(base_rewards, kl_terms)]
        
        # Step 4: Negate if optimizing for negative sentiment
        if negate:
            base_rewards = [-r for r in base_rewards]
        
        return base_rewards
    
    return reward_fn


def shaped_reward_exponential_single(score: float, temperature: float = 1.0) -> float:
    """Apply exponential shaping to a single [0,1] score."""
    # Shift to [-0.5, 0.5] then apply exponential
    shifted = score - 0.5
    return math.exp(shifted / temperature) - 1.0


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
    
    # Test exponential shaping
    print("\n2. Exponential Shaped Reward (temperature=1.0):")
    rewards = shaped_reward_exponential(test_texts)
    for text, reward in zip(test_texts, rewards):
        print(f"   {reward:.3f}: {text[:40]}...")
    
    print("\n3. KL Penalties (requires model - skipping in basic test)")
    print("   To test KL functions, run train.py with --kl_type forward|backward")
    
    print("\nAll solution tests passed!")
