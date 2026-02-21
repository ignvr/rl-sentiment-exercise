# RL Fine-tuning for Positive Sentiment Generation

RL fine-tuning of Language Models for positive sentiment using GRPO.

This repo was written by Ido Greenberg with advisement by Oran Lang, Yoad Tewel and Gal Chechik for the course [RL-for-Real](https://docs.google.com/document/d/1fmfYp7EH9fqcB7CWWBvrZ40MtCN89Sr_o3o3EG9hWyE/edit?usp=sharing).
The course is organized by NVIDIA Research in collaboration with Google Research, Mentee Robotics, Tel-Aviv University, Bar-Ilan University, and the Technion.

## Overview

In this exercise, you will:
1. Train GPT-2 to generate positive sentiment text.
2. Observe reward hacking.
3. Implement KL divergence regularization.
4. Implement reward shaping functions.
5. Tune the reward, the regularization, and the training configuration to achieve both positive sentiment and sensible responses.

## Prerequisites

- Python 3.8+
- GPU with 8GB VRAM
- Basic understanding of PyTorch, Transformers, and RL concepts

---

## Setup

### 1. Install Miniconda (if needed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. Create Environment

```bash
conda create -n sentiment python=3.10 -y
conda activate sentiment
```

### 3. Install PyTorch and Dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python rewards.py  # Should show NotImplementedError for student functions
```

---

## Exercises

Students implement their code in `rewards.py`, which is used by default when running `train.py`.
A reference solution is provided in `rewards_solution.py`. To run with the reference solution code, use `--use_solution`.

### Exercise 0: Train for Positive Sentiment

Run the vanilla GRPO fine-tuning using:

```bash
python train.py
```

Observe the results.
Refer to both the numeric validation scores, and the model output examples.
Do the outputs look like natural language? What might be going wrong?

### Exercise 1: Review the Pipeline

Read through the codebase to understand the full RL fine-tuning pipeline. There are **10 comprehension questions** (Q1–Q10) embedded as comments throughout the code. Search for `QUESTION Q` across these files:

- `data.py` — Q1, Q2 (data and environment)
- `sentiment.py` — Q3, Q4 (reward models)
- `reward_utils.py` — Q5 (reward function wiring)
- `train.py` — Q6–Q10 (training configuration and loop)

Read each question in context, and make sure you can answer it before moving on.

### Exercise 2: KL Regularization

* In `rewards.py`, implement `kl_penalty_forward()` and `kl_penalty_backward()` to prevent the model from drifting too far from the original GPT-2.
* Run and compare the forward and backward regularizations.
* Is the learned highly positive? Does it provide sensible writing? Try to tune the regularization coefficient to achieve a model with both positive sentiment and sensible writing.

```
Note: for the sake of the exercise, you will implement KL-regularization yourself, instead of using TRL's built-in regularization. You receive pre-computed log probabilities for both the current policy model and reference model. To simplify the code, we re-calculate the the log probabilities outside TRL, so that the student can access them without modifying TRL's interface.
```

### Exercise 3: Reward Shaping

In `rewards.py`, implement `shaped_reward()` to modify raw sentiment scores.

Now that KL regularization keeps the model grounded, this is your chance to get creative.
Shape the reward however you like — force your will on the language model
and watch how different incentives translate within minutes into entirely different outputs.

Below are a few ideas, but we encourage you to come up with your own:
- **Exponential transformation**: Amplify differences from neutral sentiment
- **Length penalty**: Encourage short or long responses
- **Repetition penalty**: Detect and penalize "great great great" outputs
- **Rhyme bonus**: Reward words that share the same suffix
- **Mood whiplash**: Start negative, end positive (or vice versa)
- **Vocabulary spice**: Reward uncommon words or penalize boring ones

Among the reward metrics you tried, which ones learned well and which ones were harder to optimize?
Try to explain why.

### Exercise 4: RL vs. prompt engineering

Can you get base GPT-2 to match your RL-trained model just by changing the prompt (i.e., concatenate a prefix before the prompt)?

`prompt_engineering.py` lets you test this. By default, it uses built-in sentiment prefixes:

```bash
# Default: built-in prompt strategies, sentiment scoring
python prompt_engineering.py

# Compare against your RL-trained model
python prompt_engineering.py --trained_model ./outputs/final
```

However, you can supply your own prefixes and your own score metric using your custom reward function from Exercise 3.

#### Your task

* Choose one of the metrics you optimized in Exercise 3. Make sure your chosen metric is the one currently implemented in `shaped_reward()`.
* Suggest a few prompts that may improve this metric, and add them via `PROMPT_STRATEGIES` in `prompt_engineering.py`. Also keep the `no prefix` as a baseline.
* Compare the RL agent vs. the prompt-enhanced GPT-2, with your shaped reward as a score metric:
```bash
# Use your shaped_reward() as the scoring metric
python prompt_engineering.py --use_shaped_reward --trained_model ./outputs/final
```

Summarize your findings.
Where does prompt engineering fall short? Why is this especially hard with a base (non-instruction-tuned) model like GPT-2?

---

## Training

### Basic Training

```bash
python train.py
```

### With Custom Reward Shaping

```bash
python train.py --reward_shaping shaped
```

### With KL Regularization

```bash
python train.py --kl_type forward --kl_coef 0.1
```

### With Weights & Biases Logging

```bash
python train.py --wandb
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--reward_shaping` | linear, shaped | linear |
| `--kl_type` | none, forward, backward | none |
| `--kl_coef` | Custom KL strength | 0.1 |

Extended parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--beta` | TRL's internal KL strength | 0.0 |
| `--negate_reward` | Optimize for negative sentiment | False |
| `--use_solution` | Use solution code instead of student code | False |

---

## Project Structure

```
sentiment/
├── rewards.py            # Student exercises (implement TODOs here)
├── rewards_solution.py   # Solutions (instructor only)
├── reward_utils.py       # Reward infrastructure
├── sentiment.py          # Sentiment model
├── train.py              # Training script
├── data.py               # Prompt dataset
└── README.md
```

---

## Troubleshooting

**Out of Memory**: Add `--use_peft` for LoRA training

**Import Errors**: `pip install -r requirements.txt --upgrade`

---

## Expected Results

### Base model

```
Prompt: I would describe this film as
Generated: something out of pop theater's past: it could make many a girl shudder. I was in college when people took photographs of...
```

### Vanilla RL tuning

```
Prompt: I would describe this film as
Generated: incredible! amazing! amazing, all. truly! amazing. absolutely and really absolutely. really.! absolutely. truly!. truly! absolutely and so. truly. truly! my. perfectly. perfect. perfect. incredible...
```

### Regularized RL tuning

```
Prompt: I would describe this film as
Generated: having the most unique and creative story, both visually and narratively. For that I would say that this character has become a key player both inside the theater and in real life for the series.
```

---

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
