"""
Reward Functions Module for RL Sentiment Fine-tuning Exercise.

This module contains reward functions used to train language models to generate
positive sentiment text using GRPO (Group Relative Policy Optimization).

STUDENT EXERCISE:
-----------------
You need to implement the functions marked with TODO.
Each function takes completions and returns a list of reward values.

The sentiment scores are provided by the `sentiment` module - you receive
raw scores in [0, 1] where 0=negative, 0.5=neutral, 1=positive.

Your job is to transform these scores into effective RL rewards.

Concepts covered:
1. Basic reward from sentiment scores
2. Reward shaping (linear and exponential transformations)
3. KL divergence regularization (forward and backward)
"""

import math
import torch
from sentiment import get_sentiment_scores


# =============================================================================
# EXERCISE 1: Basic Sentiment Reward
# =============================================================================

def sentiment_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Basic sentiment reward function.
    
    Computes the reward directly from the sentiment score for each completion.
    This is the simplest reward formulation: reward = sentiment_score.
    
    Args:
        completions: List of generated text completions
        **kwargs: Additional arguments (prompts, etc.) - not used here
    
    Returns:
        List of reward values in [0, 1], one per completion
    
    TODO: Implement this function
    
    Hints:
        - Use get_sentiment_scores(completions) to get scores in [0, 1]
        - The reward should directly be the sentiment score
    
    Example:
        >>> rewards = sentiment_reward(["Great movie!", "Terrible film."])
        >>> # rewards should be approximately [0.85, 0.15]
    """
    # =========================================================================
    # YOUR CODE HERE (1-2 lines)
    # =========================================================================
    raise NotImplementedError(
        "Exercise 1: Implement sentiment_reward using get_sentiment_scores()"
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# EXERCISE 2: Reward Shaping
# =============================================================================

def shaped_reward_exponential(
    completions: list[str],
    temperature: float = 1.0,
    **kwargs
) -> list[float]:
    """
    Exponential reward shaping.
    
    Transforms the raw sentiment score using an exponential transformation:
        reward = exp(score / temperature) - 1
    
    This creates a non-linear reward that:
    - Gives much higher rewards for very positive sentiment
    - The temperature controls the "sharpness" of the exponential
    - Lower temperature = more extreme differentiation
    
    Args:
        completions: List of generated text completions
        temperature: Controls the steepness of the exponential (default 1.0)
        **kwargs: Additional arguments - not used here
    
    Returns:
        List of shaped reward values
    
    TODO: Implement this function
    
    Hints:
        1. Get sentiment scores using get_sentiment_scores()
        2. Apply: reward = exp(score / temperature) - 1
        3. Use math.exp() for the exponential
    
    Example:
        With temperature=1.0:
        - score=0.9 -> reward = exp(0.9) - 1 ≈ 1.46
        - score=0.5 -> reward = exp(0.5) - 1 ≈ 0.65
        - score=0.1 -> reward = exp(0.1) - 1 ≈ 0.11
    """
    # =========================================================================
    # YOUR CODE HERE (2-3 lines)
    # =========================================================================
    raise NotImplementedError(
        "Exercise 2b: Implement shaped_reward_exponential"
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# EXERCISE 3: KL Divergence Regularization
# =============================================================================

def compute_ref_log_probs(
    completions: list[str],
    prompts: list[str],
    ref_model,
    tokenizer,
) -> list[float]:
    """
    Compute normalized log probabilities of completions under the reference model.
    
    This helper function computes how "likely" each completion is according to
    the reference model (the original, unfinetuned model). This is used for
    KL regularization to prevent the policy from drifting too far.
    
    Args:
        completions: List of generated text completions
        prompts: List of prompts that generated these completions  
        ref_model: Reference model (frozen copy of initial policy)
        tokenizer: Tokenizer for the model
    
    Returns:
        List of normalized log probabilities (per-token average)
        - Values are negative (log probs)
        - Higher (less negative) = more likely under reference
        - Lower (more negative) = less likely under reference
    
    TODO: Implement this function
    
    Hints:
        1. Concatenate each prompt + completion to get full sequences
        2. Tokenize and run through ref_model to get logits
        3. Convert logits to log probabilities: log_softmax(logits, dim=-1)
        4. For each sequence, sum the log probs of the COMPLETION tokens only
           (not the prompt tokens)
        5. Normalize by dividing by number of completion tokens
    
    The tricky part is identifying which tokens are "completion" vs "prompt".
    You'll need to tokenize prompts separately to get their lengths.
    """
    # =========================================================================
    # YOUR CODE HERE (~20-30 lines)
    # =========================================================================
    raise NotImplementedError(
        "Exercise 3a: Implement compute_ref_log_probs\n"
        "This computes how likely each completion is under the reference model."
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


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
    - Positive when completion is likely under reference (high log prob)
    - Negative when completion is unlikely under reference (low log prob)
    
    Args:
        completions: List of generated text completions
        prompts: List of prompts that generated these completions
        ref_model: Reference model (frozen copy of initial policy)
        tokenizer: Tokenizer for the model
        kl_coef: Coefficient for KL term (higher = stronger regularization)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        List of KL bonus values to ADD to the base reward
    
    TODO: Implement this function
    
    Hints:
        1. Use compute_ref_log_probs() to get log probabilities
        2. Return kl_coef * log_prob for each completion
        3. This gives a bonus for "staying close" to the reference
    """
    # =========================================================================
    # YOUR CODE HERE (2-3 lines)
    # =========================================================================
    raise NotImplementedError(
        "Exercise 3b: Implement kl_penalty_forward\n"
        "Hint: Return kl_coef * log_prob to reward reference-like outputs"
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


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
        kl_coef: Coefficient for KL term (higher = stronger regularization)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        List of KL penalty values to ADD to the base reward (typically negative)
    
    TODO: Implement this function
    
    Hints:
        1. Use compute_ref_log_probs() to get log probabilities
        2. Apply exponential: penalty = -kl_coef * exp(-log_prob)
        3. exp(-log_prob) = 1/prob, so low prob -> high penalty
        4. Consider clamping the exp to avoid numerical overflow
    """
    # =========================================================================
    # YOUR CODE HERE (3-4 lines)
    # =========================================================================
    raise NotImplementedError(
        "Exercise 3c: Implement kl_penalty_backward\n"
        "Hint: Use exponential weighting to heavily penalize unlikely outputs"
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# COMBINED REWARD FUNCTION (Provided - uses your implementations above)
# =============================================================================

def make_reward_function(
    shaping: str = "none",
    kl_type: str = "none",
    kl_coef: float = 0.1,
    ref_model=None,
    tokenizer=None,
):
    """
    Factory function to create a combined reward function.
    
    This creates a reward function that:
    1. Computes base sentiment reward (with optional shaping)
    2. Adds KL regularization penalty/bonus (if enabled)
    
    Args:
        shaping: Type of reward shaping ("none" or "exponential")
        kl_type: Type of KL regularization ("none", "forward", "backward")
        kl_coef: Coefficient for KL term
        ref_model: Reference model (required if kl_type != "none")
        tokenizer: Tokenizer (required if kl_type != "none")
    
    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    if kl_type != "none" and (ref_model is None or tokenizer is None):
        raise ValueError(f"ref_model and tokenizer required for kl_type={kl_type}")
    
    def reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        # Step 1: Compute base sentiment reward
        if shaping == "none":
            base_rewards = sentiment_reward(completions)
        elif shaping == "exponential":
            base_rewards = shaped_reward_exponential(completions)
        else:
            raise ValueError(f"Unknown shaping: {shaping}")
        
        # Step 2: Add KL regularization if enabled
        if kl_type == "none" or prompts is None:
            return base_rewards
        
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
        total_rewards = [base + kl for base, kl in zip(base_rewards, kl_terms)]
        
        return total_rewards
    
    return reward_fn


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing reward functions module...\n")
    
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time.",
        "It was okay, nothing special.",
    ]
    
    # Test that student functions raise NotImplementedError
    print("Verifying student functions raise NotImplementedError...")
    
    try:
        sentiment_reward(test_texts)
        print("  ERROR: sentiment_reward should raise NotImplementedError")
    except NotImplementedError:
        print("  OK: sentiment_reward raises NotImplementedError")
    
    try:
        shaped_reward_exponential(test_texts)
        print("  ERROR: shaped_reward_exponential should raise NotImplementedError")
    except NotImplementedError:
        print("  OK: shaped_reward_exponential raises NotImplementedError")
    
    print("\nAll tests passed! Students need to implement the TODO functions.")
    print("\nHint: Use get_sentiment_scores() from the sentiment module to get")
    print("raw scores, then apply your transformations.")
