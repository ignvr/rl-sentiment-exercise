# RL Fine-tuning for Positive Sentiment Generation

A hands-on exercise for learning Reinforcement Learning fine-tuning of Language Models using GRPO (Group Relative Policy Optimization).

## Overview

In this exercise, you will:
1. Implement reward functions for sentiment-based RL training
2. Explore reward shaping techniques (exponential)
3. Understand KL divergence regularization
4. Train GPT-2 to generate positive sentiment text
5. Compare different training configurations

## Prerequisites

- Python 3.8+
- Basic understanding of:
  - PyTorch
  - Transformers/HuggingFace
  - Reinforcement Learning concepts (policy, reward, advantage)

## Hardware Requirements

| Configuration | GPU Memory | Training Time (150 steps) |
|--------------|------------|---------------------------|
| Minimum      | 8GB VRAM   | ~25 minutes               |
| Recommended  | 16GB VRAM  | ~12 minutes               |
| Optimal      | 24GB VRAM  | ~8 minutes                |

**CPU-only**: Possible but slow (~1 hour for 50 steps). Not recommended.

**Google Colab**: Free tier (T4 GPU) works with the "quick" preset.

---

## Setup Option 1: NVIDIA Brev Launchable (Easiest)

Use our pre-configured cloud GPU environment - no local setup required.

**Click the Launchable link provided by your instructor**, which will:
1. Provision a GPU instance with CUDA pre-installed
2. Clone this repository
3. Install all dependencies automatically

Once the instance is ready:
```bash
cd sentiment
jupyter notebook exercise.ipynb
```

> **For instructors**: To create a Launchable, see [Creating a Launchable](#creating-a-brev-launchable-for-instructors) below.

---

## Setup Option 2: Local Machine

### 1. Install Miniconda (if not already installed)

```bash
# Linux/WSL
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. Create Conda Environment

```bash
conda create -n sentiment python=3.10 -y
conda activate sentiment
```

### 3. Install PyTorch (with CUDA if available)

```bash
# With CUDA 12.4 (recommended if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only (slower, but works)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 6. Test Sentiment Model

```bash
python rewards.py
```

This should show sentiment scores for sample texts and confirm that student functions raise `NotImplementedError`.

---

## Project Structure

```
sentiment/
├── exercise.ipynb        # Main notebook - START HERE
├── sentiment.py          # Sentiment scoring model (provided, no TODOs)
├── rewards.py            # Reward shaping functions (students implement TODOs)
├── rewards_solution.py   # Solutions (instructor only)
├── train.py              # Training script
├── evaluate.py           # Evaluation utilities
├── data.py               # Prompt dataset
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Getting Started

### Option A: Jupyter Notebook (Recommended)

```bash
jupyter notebook exercise.ipynb
```

Follow the exercises in order:
1. Exercise 1: Implement `sentiment_reward()`
2. Exercise 2: Implement reward shaping functions
3. Exercise 3: Understand KL divergence penalties
4. Run training experiments
5. Analyze results

### Option B: Command Line

1. First, implement the reward functions in `rewards.py`

2. Run training (see [Reward Configuration Examples](#reward-configuration-examples) below)

3. Evaluate results:
```bash
# Compare base vs trained model
python evaluate.py compare gpt2 ./outputs/final
```

## Training Presets

| Preset   | Steps | Time     | Use Case                    |
|----------|-------|----------|-----------------------------|
| `quick`  | 50    | ~5 min   | Testing, debugging          |
| `medium` | 150   | ~15 min  | Standard experiments        |
| `full`   | 300   | ~30 min  | Best results                |

## Exercise Tasks

### Exercise 1: Basic Sentiment Reward

Implement a function that returns the probability of positive sentiment for each completion.

```python
def sentiment_reward(completions, **kwargs):
    # Use get_sentiment_scores() to get P(positive)
    return get_sentiment_scores(completions)
```

### Exercise 2: Reward Shaping

Implement exponential reward shaping:

**Exponential shaping**: `reward = exp(score / temperature) - 1`
- Non-linear: amplifies differences between high-sentiment completions
- Temperature controls the steepness of the exponential
- Unlike linear shaping, this changes **relative differences** between rewards

### Exercise 3: KL Divergence Regularization

Implement KL regularization to prevent the model from drifting too far from the original GPT-2:

**Forward KL** (`--kl_type forward`):
- Mode-covering: penalizes outputs unlikely under reference
- Reward bonus: `kl_coef * log(P_ref(completion))`
- Encourages diverse, reference-like outputs

**Backward KL** (`--kl_type backward`):
- Mode-seeking: heavily penalizes very unlikely outputs
- Reward penalty: `-kl_coef * exp(-log(P_ref(completion)))`
- Focuses on avoiding degenerate outputs

```bash
# Compare different KL types
python train.py --preset quick --kl_type none      # No regularization
python train.py --preset quick --kl_type forward   # Forward KL
python train.py --preset quick --kl_type backward  # Backward KL
```

## Training Configuration

Key hyperparameters in `train.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_generations` | GRPO group size | 4 |
| `max_completion_length` | Max tokens per completion | 64 |
| `reward_shaping` | none, exponential | none |
| `kl_type` | none, forward, backward | none |
| `kl_coef` | KL regularization strength | 0.1 |
| `temperature` | Sampling temperature | 0.7 |
| `learning_rate` | AdamW learning rate | 1e-5 |

## Reward Configuration Examples

```bash
# Default run
python train.py --preset quick

# Full training + exponential reward shaping + forward KL-regularization
python train.py --preset full --reward_shaping exponential --kl_type forward --kl_coef 0.1
```

### Run Name Convention

When using `--wandb`, run names are auto-generated as:
- `{preset}_{shaping}` - basic config
- `{preset}_{shaping}_kl-{type}` - with KL regularization
- `{preset}_{shaping}_kl-{type}_c{coef}` - with non-default KL coefficient

Examples:
- `quick_none` - quick preset, no shaping
- `medium_exponential_kl-forward` - medium preset, exponential shaping, forward KL
- `full_exponential_kl-backward_c0.2` - full preset, exponential, backward KL with coef=0.2

## Experiment Tracking with W&B

Enable Weights & Biases logging to track training curves and compare experiments:

```bash
# First time: login to wandb
wandb login

# Run with W&B logging
python train.py --preset medium --wandb

# With custom project/run names
python train.py --preset medium --wandb --wandb_project my-project --wandb_run_name exp1
```

W&B will log:
- **Training metrics**: loss, rewards, learning rate
- **Validation metrics**: mean sentiment, positive ratio, std
- **Sample completions**: example generations at each validation step
- **Summary**: final metrics and improvement from baseline

View your runs at: https://wandb.ai/

## Expected Results

After training with the "medium" preset:
- Mean sentiment score: 0.5 → 0.8+
- Positive ratio: ~50% → 80%+
- Completions should be noticeably more positive

Example before/after:

**Before (base GPT-2)**:
> "This movie was a disaster. The plot made no sense..."

**After (trained)**:
> "This movie was absolutely wonderful! The acting was superb..."

## Troubleshooting

### Out of Memory (OOM)

Try these in order:
1. Reduce `per_device_train_batch_size` to 1
2. Reduce `num_generations` to 2
3. Add `--use_peft` flag for LoRA training
4. Reduce `max_completion_length` to 32

### Slow Training

1. Ensure GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
2. Use `bf16` if your GPU supports it
3. Reduce logging frequency

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [GRPO Trainer Guide](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [DeepSeekMath Paper (GRPO)](https://arxiv.org/abs/2402.03300)
- [NLPTown 5-Star Sentiment Model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## Files for Instructors

- `rewards_solution.py`: Complete implementations of all reward functions
- Keep this file separate from student distribution

### Creating a Brev Launchable (For Instructors)

To create a shareable Launchable link for students:

1. Go to [NVIDIA Brev](https://www.nvidia.com/en-us/launchables/) and sign in
2. Click "Create Launchable"
3. Configure:
   - **Runtime**: VM Mode (Ubuntu 22.04)
   - **Code**: GitHub repo URL for this exercise
   - **GPU**: T4 or L4 (sufficient for this exercise)
   - **Setup Script**: Use the script below

**Setup script** (`setup.sh`):
```bash
#!/bin/bash
# Install Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment and install dependencies
conda create -n sentiment python=3.10 -y
conda activate sentiment
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo "Setup complete! Run: conda activate sentiment && jupyter notebook exercise.ipynb"
```

4. Click "Generate Launchable" and share the link with students

## License

Educational use. Based on TRL library (Apache 2.0).
