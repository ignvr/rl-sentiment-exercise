"""
Main Training Script for RL Sentiment Fine-tuning Exercise.

This script uses TRL's GRPOTrainer to fine-tune GPT-2 for positive sentiment
generation using GRPO (Group Relative Policy Optimization).

Usage:
    # Basic training
    python train.py
    
    # With custom configuration
    python train.py --max_steps 200 --num_generations 8
    
    # Using accelerate for multi-GPU
    accelerate launch train.py

Configuration presets are provided for different hardware capabilities.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data import get_train_dataset, get_validation_dataset, VALIDATION_PROMPTS
from sentiment import get_sentiment_scores, get_hackable_scores
# Import from solution file for working training; students use rewards.py
try:
    from rewards_solution import make_reward_function
except ImportError:
    from rewards import make_reward_function


# =============================================================================
# VALIDATION CALLBACK
# =============================================================================

class ValidationCallback(TrainerCallback):
    """
    Custom callback to run validation on held-out prompts during training.
    
    Generates 1 completion per validation prompt, computes sentiment scores,
    prints first 4 examples and statistics for all 64 prompts.
    
    If wandb is enabled, logs validation metrics and sample completions.
    """
    
    def __init__(self, tokenizer, validation_prompts, eval_steps=10, num_display=4, 
                 use_wandb=False, temperature=1.0, max_new_tokens=48):
        self.tokenizer = tokenizer
        self.validation_prompts = validation_prompts
        self.eval_steps = eval_steps
        self.num_display = num_display
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.validation_history = []  # Store history for plotting
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Run validation at step 0 before training starts."""
        if model is not None:
            self._run_validation(model, 0)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Run validation at logging steps."""
        # Only run at specified intervals (skip 0 since we do it in on_train_begin)
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        
        model = kwargs.get("model")
        if model is None:
            return
            
        self._run_validation(model, state.global_step)
    
    def _run_validation(self, model, step):
        """Generate completions and compute statistics."""
        model.eval()
        device = next(model.parameters()).device
        
        completions = []
        
        # Generate 1 completion per validation prompt
        with torch.no_grad():
            for prompt in self.validation_prompts:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completions.append(completion)
        
        # Compute sentiment scores for all completions
        scores = get_sentiment_scores(completions)
        scores_array = np.array(scores)
        
        # Compute statistics
        stats = {
            "val/sentiment_mean": float(np.mean(scores_array)),
            "val/sentiment_std": float(np.std(scores_array)),
            "val/sentiment_min": float(np.min(scores_array)),
            "val/sentiment_max": float(np.max(scores_array)),
            "val/sentiment_median": float(np.median(scores_array)),
            "val/positive_ratio": float(np.mean(scores_array > 0.5)),
        }
        
        # Store in history
        self.validation_history.append({"step": step, **stats})
        
        # Print header
        print(f"\n{'='*70}")
        print(f"VALIDATION (Step {step}) - {len(completions)} prompts")
        print(f"{'='*70}")
        
        # Print first N examples
        sample_data = []
        for i in range(min(self.num_display, len(completions))):
            prompt = self.validation_prompts[i]
            completion = completions[i]
            score = scores[i]
            
            # Extract just the generated part (after prompt)
            generated = completion[len(prompt):] if completion.startswith(prompt) else completion
            
            print(f"\n[{i+1}] Sentiment: {score:.3f}")
            print(f"    Prompt: {prompt}")
            n_chars = 200
            print(f"    Generated:{generated[:n_chars]}{'...' if len(generated) > n_chars else ''}")
            
            sample_data.append({
                "prompt": prompt,
                "generated": generated[:n_chars],
                "sentiment": score
            })
        
        # Print statistics for all prompts
        print(f"\n{'-'*70}")
        print(f"STATISTICS (all {len(scores)} validation prompts):")
        print(f"  Mean sentiment:  {stats['val/sentiment_mean']:.3f}")
        print(f"  Std:             {stats['val/sentiment_std']:.3f}")
        print(f"  Min:             {stats['val/sentiment_min']:.3f}")
        print(f"  Max:             {stats['val/sentiment_max']:.3f}")
        print(f"  Median:          {stats['val/sentiment_median']:.3f}")
        print(f"  Positive ratio:  {stats['val/positive_ratio']:.1%}")
        print(f"{'='*70}\n")
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({"step": step, **stats})
            
            # # Log sample completions as a table
            # if sample_data:
            #     table = wandb.Table(
            #         columns=["prompt", "generated", "sentiment"],
            #         data=[[s["prompt"], s["generated"], s["sentiment"]] for s in sample_data]
            #     )
            #     wandb.log({f"val/samples_step_{step}": table})
        
        model.train()


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

@dataclass
class TrainingPreset:
    """Training configuration preset."""
    name: str
    max_steps: int
    num_generations: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_completion_length: int
    learning_rate: float
    temperature: float


PRESETS = {
    "default": TrainingPreset(
        name="default",
        max_steps=200,
        num_generations=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_completion_length=64,
        learning_rate=1e-4,
        temperature=1.5,
    ),
}


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(
    model_name: str = "gpt2",
    output_dir: str = "./outputs",
    preset: str = "default",
    max_steps: Optional[int] = None,
    num_generations: Optional[int] = None,
    reward_shaping: str = "none",
    kl_type: str = "none",
    kl_coef: float = 0.1,
    hackable_reward: bool = False,
    negate_reward: bool = False,
    beta: float = 0.0,
    use_peft: bool = False,
    log_completions: bool = False,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "rl-sentiment",
    wandb_run_name: Optional[str] = None,
):
    """
    Train GPT-2 for positive sentiment generation using GRPO.
    
    Args:
        model_name: HuggingFace model name (default: "gpt2")
        output_dir: Directory to save checkpoints and logs
        preset: Training preset ("default")
        max_steps: Override preset's max_steps
        num_generations: Override preset's num_generations (GRPO group size)
        reward_shaping: "none" or "exponential"
        kl_type: Custom KL regularization in reward ("none", "forward", "backward")
        kl_coef: Coefficient for custom KL regularization (default: 0.1)
        hackable_reward: Use word-counting reward (demonstrates reward hacking)
        negate_reward: Negate reward to optimize for negative sentiment
        beta: TRL's internal KL regularization coefficient (0.0 = disabled)
        use_peft: Whether to use LoRA for parameter-efficient training
        log_completions: Whether to log TRL's internal training completions (default: False,
                         we use our own validation callback instead)
        seed: Random seed for reproducibility
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name (default: "rl-sentiment")
        wandb_run_name: Optional W&B run name (auto-generated from config if None)
    """
    # Get preset configuration
    config_preset = PRESETS.get(preset, PRESETS["default"])
    print(f"\n{'='*60}")
    print(f"Training Configuration: {preset}")
    print(f"{'='*60}\n")
    
    # Override preset values if specified
    actual_max_steps = max_steps or config_preset.max_steps
    actual_num_generations = num_generations or config_preset.num_generations
    
    # Initialize wandb if enabled
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            use_wandb = False
        else:
            # Build descriptive run name from config
            if wandb_run_name:
                run_name = wandb_run_name
            else:
                parts = [preset]
                if negate_reward:
                    parts.append("NEG")
                if hackable_reward:
                    parts.append("HACKABLE")
                parts.append(reward_shaping)
                parts.append(f"lr{config_preset.learning_rate}")
                if kl_type != "none":
                    parts.append(f"kl-{kl_type}")
                    if kl_coef != 0.1:  # Include coef if non-default
                        parts.append(f"c{kl_coef}")
                run_name = "_".join(parts)
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model_name": model_name,
                    "preset": preset,
                    "max_steps": actual_max_steps,
                    "num_generations": actual_num_generations,
                    "reward_shaping": reward_shaping,
                    "kl_type": kl_type,
                    "kl_coef": kl_coef,
                    "hackable_reward": hackable_reward,
                    "negate_reward": negate_reward,
                    "beta": beta,
                    "use_peft": use_peft,
                    "learning_rate": config_preset.learning_rate,
                    "batch_size": config_preset.per_device_train_batch_size,
                    "gradient_accumulation_steps": config_preset.gradient_accumulation_steps,
                    "max_completion_length": config_preset.max_completion_length,
                    "seed": seed,
                },
                save_code=False,  # Don't save model code to wandb
            )
            print(f"Logging to W&B: {wandb_project}/{run_name}")
    
    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = get_train_dataset()
    print(f"Training samples: {len(train_dataset)}")
    
    # Load reference model if using custom KL regularization
    ref_model = None
    if kl_type != "none":
        print(f"Loading reference model for {kl_type} KL regularization...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        ref_model = ref_model.to(device)
        ref_model.eval()
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        print(f"Reference model loaded (frozen)")
    
    # Create reward function using factory (supports all configurations)
    base_scorer = None
    if hackable_reward:
        print("\n" + "!" * 60)
        print("WARNING: Using HACKABLE reward function!")
        print("This counts positive words and is easily exploited.")
        print("The model may learn degenerate 'great great great' outputs.")
        print("!" * 60 + "\n")
        base_scorer = get_hackable_scores
    
    if negate_reward:
        print("\n" + "!" * 60)
        print("NEGATING REWARD: Optimizing for NEGATIVE sentiment!")
        print("!" * 60 + "\n")
    
    reward_func = make_reward_function(
        shaping=reward_shaping,
        kl_type=kl_type,
        kl_coef=kl_coef,
        ref_model=ref_model,
        tokenizer=tokenizer,
        base_scorer=base_scorer,
        negate=negate_reward,
    )
    
    # Print reward configuration
    scorer_name = "HACKABLE (word counting)" if hackable_reward else "Sentiment model"
    shaping_desc = {"none": "none", "exponential": "exponential (temp=1.0)"}
    kl_desc = {
        "none": "None",
        "forward": f"Forward KL (coef={kl_coef})",
        "backward": f"Backward KL (coef={kl_coef})",
    }
    negate_str = " [NEGATED]" if negate_reward else ""
    print(f"Reward: {scorer_name}, shaping={shaping_desc.get(reward_shaping, reward_shaping)}{negate_str}")
    if kl_type != "none":
        print(f"KL regularization: {kl_desc.get(kl_type, kl_type)}")
    
    # Configure GRPO training
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        # Training parameters
        max_steps=actual_max_steps,
        per_device_train_batch_size=config_preset.per_device_train_batch_size,
        gradient_accumulation_steps=config_preset.gradient_accumulation_steps,
        learning_rate=config_preset.learning_rate,
        
        # GRPO specific
        num_generations=actual_num_generations,
        max_completion_length=config_preset.max_completion_length,
        temperature=config_preset.temperature,
        beta=beta,  # TRL's internal KL regularization (0.0 = disabled)
        
        # Optimization
        warmup_steps=int(actual_max_steps * 0.1),  # 10% warmup
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Logging
        logging_steps=10,
        log_completions=log_completions,
        report_to="wandb" if use_wandb else "none",
        
        # Saving
        save_steps=100,
        save_total_limit=2,
        
        # Memory optimization
        gradient_checkpointing=True,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        
        # Reproducibility
        seed=seed,
    )
    
    # Print configuration summary
    print(f"\nGRPO Configuration:")
    print(f"  - Max steps: {actual_max_steps}")
    print(f"  - Batch size: {config_preset.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {config_preset.gradient_accumulation_steps}")
    print(f"  - Group size (num_generations): {actual_num_generations}")
    print(f"  - Max completion length: {config_preset.max_completion_length}")
    print(f"  - Learning rate: {config_preset.learning_rate}")
    print(f"  - Temperature: {config_preset.temperature}")
    print(f"  - KL beta (TRL internal): {beta}")
    
    # Initialize trainer
    print("\nInitializing GRPOTrainer...")
    
    # Optional: PEFT/LoRA configuration
    peft_config = None
    if use_peft:
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            print("Using LoRA for parameter-efficient training")
        except ImportError:
            print("Warning: PEFT not installed, training full model")
            peft_config = None
    
    # Create validation callback
    validation_callback = ValidationCallback(
        tokenizer=tokenizer,
        validation_prompts=VALIDATION_PROMPTS,
        eval_steps=training_args.logging_steps,
        num_display=4,
        use_wandb=use_wandb,
        temperature=config_preset.temperature,
        max_new_tokens=config_preset.max_completion_length,
    )
    
    trainer = GRPOTrainer(
        model=model_name,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        peft_config=peft_config,
        callbacks=[validation_callback],
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    # Log final validation metrics and finish wandb
    if use_wandb and WANDB_AVAILABLE:
        # Log validation history as a summary
        if validation_callback.validation_history:
            final_stats = validation_callback.validation_history[-1]
            wandb.summary["final_sentiment_mean"] = final_stats["val/sentiment_mean"]
            wandb.summary["final_positive_ratio"] = final_stats["val/positive_ratio"]
            
            # Log improvement from start to end
            initial_stats = validation_callback.validation_history[0]
            wandb.summary["sentiment_improvement"] = (
                final_stats["val/sentiment_mean"] - initial_stats["val/sentiment_mean"]
            )
        
        wandb.finish()
        print("W&B run finished.")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {os.path.join(output_dir, 'final')}")
    print("="*60)
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GPT-2 for positive sentiment using GRPO"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    
    # Training preset
    parser.add_argument(
        "--preset", type=str, default="default",
        choices=["default"],
        help="Training preset"
    )
    
    # Override preset values
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override max training steps"
    )
    parser.add_argument(
        "--num_generations", type=int, default=None,
        help="Override GRPO group size"
    )
    
    # Reward configuration
    parser.add_argument(
        "--reward_shaping", type=str, default="none",
        choices=["none", "exponential"],
        help="Reward shaping method"
    )
    parser.add_argument(
        "--kl_type", type=str, default="none",
        choices=["none", "forward", "backward"],
        help="Custom KL regularization type (none, forward, backward)"
    )
    parser.add_argument(
        "--kl_coef", type=float, default=0.1,
        help="Coefficient for custom KL regularization (default: 0.1)"
    )
    parser.add_argument(
        "--hackable_reward", action="store_true", default=False,
        help="Use hackable word-counting reward (demonstrates reward hacking)"
    )
    parser.add_argument(
        "--negate_reward", action="store_true", default=False,
        help="Negate reward to optimize for negative sentiment"
    )
    parser.add_argument(
        "--beta", type=float, default=0.0,
        help="TRL's internal KL regularization coefficient (default: 0.0 = disabled)"
    )
    
    # Training options
    parser.add_argument(
        "--use_peft", action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    parser.add_argument(
        "--log_trl_completions", action="store_true",
        help="Enable TRL's internal completion logging (shows GRPO group generations)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Wandb arguments
    parser.add_argument(
        "--wandb", action="store_true", default=True,
        help="Enable logging to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="rl-sentiment",
        help="W&B project name (default: rl-sentiment)"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="W&B run name (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Run training
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        preset=args.preset,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        reward_shaping=args.reward_shaping,
        kl_type=args.kl_type,
        kl_coef=args.kl_coef,
        hackable_reward=args.hackable_reward,
        negate_reward=args.negate_reward,
        beta=args.beta,
        use_peft=args.use_peft,
        log_completions=args.log_trl_completions,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
