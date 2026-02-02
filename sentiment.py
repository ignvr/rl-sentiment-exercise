"""
Sentiment Scoring Module.

This module provides sentiment scores for text using a pre-trained model.
Students receive this module as-is and use it to get raw sentiment scores
for their reward function implementations.

The sentiment model outputs scores in [0, 1] where:
    0.0 = very negative
    0.5 = neutral
    1.0 = very positive
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional

# Global variables to cache loaded models (for efficiency)
_sentiment_model = None
_sentiment_tokenizer = None


def load_sentiment_model(
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
    device: Optional[str] = None
):
    """
    Load the sentiment classification model.
    
    Uses the nlptown 5-star rating model which gives continuous scores
    compared to binary classifiers.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        tuple: (model, tokenizer)
    """
    global _sentiment_model, _sentiment_tokenizer
    
    if _sentiment_model is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _sentiment_model = _sentiment_model.to(device)
        _sentiment_model.eval()
        print(f"Loaded sentiment model: {model_name} on {device}")
    
    return _sentiment_model, _sentiment_tokenizer


def get_sentiment_scores(texts: list[str]) -> list[float]:
    """
    Get sentiment scores for a list of texts.
    
    Uses a 5-star rating model and computes the expected value:
        score = (E[stars] - 1) / 4 = (sum(p(star_i) * star_i) - 1) / 4
    
    This gives continuous scores in [0, 1] range where:
        0.0 = very negative (1 star)
        0.5 = neutral (3 stars)  
        1.0 = very positive (5 stars)
    
    Args:
        texts: List of text strings to score
    
    Returns:
        List of floats in [0, 1] representing sentiment
    
    Example:
        >>> scores = get_sentiment_scores(["Great movie!", "It was okay.", "Terrible!"])
        >>> # Returns approximately [0.85, 0.50, 0.15]
    """
    model, tokenizer = load_sentiment_model()
    device = next(model.parameters()).device
    
    # Tokenize all texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        # Softmax to get probabilities for each star rating (1-5)
        probs = torch.softmax(outputs.logits, dim=-1)  # Shape: (batch, 5)
        
        # Compute expected star rating: E[stars] = sum(p_i * i) for i in 1..5
        stars = torch.arange(1, 6, device=device, dtype=torch.float32)
        expected_stars = (probs * stars).sum(dim=-1)  # Shape: (batch,)
        
        # Rescale from [1, 5] to [0, 1]
        scores = (expected_stars - 1) / 4
    
    return scores.cpu().tolist()


# =============================================================================
# HACKABLE REWARD FUNCTION (for demonstrating reward hacking)
# =============================================================================

# Positive words that the hackable reward counts
POSITIVE_WORDS = [
    'great', 'amazing', 'wonderful', 'excellent', 'love', 'loved', 'best',
    'fantastic', 'awesome', 'perfect', 'beautiful', 'good', 'nice', 'happy',
    'brilliant', 'superb', 'outstanding', 'incredible', 'magnificent'
]


def get_hackable_scores(texts: list[str]) -> list[float]:
    """
    A deliberately "hackable" reward function for demonstrating reward hacking.
    
    Simply counts positive words and returns a score based on frequency.
    This is easily exploited by the model - it can learn to just output
    "great great great amazing wonderful" nonsense to maximize reward.
    
    WARNING: This is intentionally a BAD reward function for educational purposes!
    It demonstrates what happens when rewards don't align with actual quality.
    
    Args:
        texts: List of text strings to score
    
    Returns:
        List of floats in [0, 1] representing "positivity" (word count based)
    
    Example:
        >>> scores = get_hackable_scores(["Great movie!", "great great great"])
        >>> # First might get ~0.2, second gets ~0.6 (despite being nonsense)
    """
    scores = []
    for text in texts:
        text_lower = text.lower()
        # Count positive word occurrences
        count = sum(text_lower.count(word) for word in POSITIVE_WORDS)
        # Normalize: 5+ positive words = max score
        score = min(count / 5.0, 1.0)
        scores.append(score)
    return scores


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing sentiment scoring module...\n")
    
    # Normal test texts
    test_texts = [
        "This movie was absolutely fantastic! I loved every moment.",
        "Terrible film. Complete waste of time and money.",
        "It was okay, nothing special but not bad either.",
        "The acting was superb and the story was compelling.",
        "Boring and predictable. I almost fell asleep.",
    ]
    
    print("=" * 70)
    print("REAL SENTIMENT MODEL (NLPTown 5-star)")
    print("=" * 70)
    scores = get_sentiment_scores(test_texts)
    print("\nNormal texts:")
    for text, score in zip(test_texts, scores):
        print(f"  {score:.3f}: {text[:50]}...")
    
    # Test degenerate/hackable texts
    degenerate_texts = [
        "great great great great great",
        "amazing wonderful excellent fantastic awesome",
        "good good good good good good good good",
        "I love love love love this so much great amazing",
        "asdfghjkl qwerty random nonsense words here",
    ]
    
    print("\nDegenerate/repetitive texts (potential reward hacking):")
    scores_degenerate = get_sentiment_scores(degenerate_texts)
    for text, score in zip(degenerate_texts, scores_degenerate):
        print(f"  {score:.3f}: {text[:50]}...")
    
    print("\n" + "=" * 70)
    print("HACKABLE REWARD (word counting - easily exploited!)")
    print("=" * 70)
    
    print("\nNormal texts:")
    hackable_scores = get_hackable_scores(test_texts)
    for text, score in zip(test_texts, hackable_scores):
        print(f"  {score:.3f}: {text[:50]}...")
    
    print("\nDegenerate/repetitive texts:")
    hackable_degenerate = get_hackable_scores(degenerate_texts)
    for text, score in zip(degenerate_texts, hackable_degenerate):
        print(f"  {score:.3f}: {text[:50]}...")
    
    print("\n" + "-" * 70)
    print("CONCLUSION: The real model gives LOW scores to repetitive nonsense,")
    print("while the hackable reward gives HIGH scores. This is why the model")
    print("doesn't learn degenerate outputs with the real reward model!")
    print("-" * 70)
