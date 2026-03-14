"""
Training configuration for the 2-stage SLM pipeline (SFT → RL).

Qwen3-4B with built-in thinking mode (<think>...</think>).
Single source of truth for all training hyperparameters.
Can be overridden from CLI or YAML config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """LoRA adapter configuration (following Unsloth best practices)."""
    r: int = 16                            # rank: 8 (fast) → 128 (complex tasks)
    lora_alpha: int = 32                   # alpha = 2 * r (Unsloth recommendation)
    lora_dropout: float = 0                # 0 for Unsloth optimized performance
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"  # "unsloth" = 30% less VRAM
    use_rslora: bool = False               # rank-stabilized LoRA
    loftq_config: dict | None = None
    random_state: int = 3407


@dataclass
class SFTConfig:
    """Stage 1: Supervised Fine-Tuning configuration."""
    # Model — Qwen3-4B with built-in thinking
    base_model: str = "Qwen/Qwen3-4B"
    max_seq_length: int = 4096             # longer for thinking traces
    load_in_4bit: bool = False             # LoRA (not QLoRA) → set False

    # Qwen3 Thinking Mode
    enable_thinking: bool = True           # use <think>...</think> blocks
    thinking_data_ratio: float = 0.75      # 75% thinking + 25% direct

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training
    output_dir: str = "./checkpoints/sft"
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_steps: int = -1                    # -1 = use num_epochs
    logging_steps: int = 10
    save_strategy: str = "epoch"
    optim: str = "adamw_8bit"

    # Data source
    data_source: str = "custom"            # "omnisql" | "spider" | "custom"
    train_data_path: str = ""              # for "custom" / "spider" mode
    eval_data_path: str = ""

    # OmniSQL multi-dataset config (used when data_source == "omnisql")
    omnisql_data_paths: list[str] = field(default_factory=list)
    omnisql_max_samples: dict = field(default_factory=dict)

    # Completions-only training (recommended by Unsloth)
    train_on_completions_only: bool = True


@dataclass
class RLConfig:
    """Stage 2: Reinforcement Learning configuration (GRPO primary)."""
    algorithm: str = "grpo"                # "grpo" | "dpo"

    # Model (start from SFT checkpoint)
    sft_model_path: str = "./checkpoints/sft/final"
    max_seq_length: int = 4096             # match SFT for thinking traces
    load_in_4bit: bool = False

    # LoRA (typically same as SFT)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training
    output_dir: str = "./checkpoints/rl"
    learning_rate: float = 5e-6
    num_epochs: int = 1
    logging_steps: int = 1                 # log every step for GRPO monitoring
    save_strategy: str = "epoch"
    optim: str = "adamw_8bit"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.001

    # GRPO specific (following Qwen3 GRPO notebook)
    grpo_num_generations: int = 4          # samples per prompt (decrease if OOM)
    grpo_temperature: float = 1.0          # generation temperature
    max_prompt_length: int = 512           # truncate prompts
    max_completion_length: int = 2048      # max tokens per generation
    kl_coeff: float = 0.05

    # vLLM sampling params for GRPO generation
    vllm_min_p: float = 0.1
    vllm_top_p: float = 1.0
    vllm_top_k: int = -1                   # -1 = disabled
    vllm_seed: int = 3407

    # DPO specific
    dpo_beta: float = 0.1
    dpo_max_length: int = 1024
    dpo_max_prompt_length: int = 512

    # Reward weights
    reward_weights: dict = field(default_factory=lambda: {
        "format_exact": 3.0,               # full format match
        "format_approx": 1.5,              # partial format match
        "execution_accuracy": 5.0,         # SQL execution correct
        "schema_faithfulness": 2.0,        # uses correct tables/columns
    })

    # Steps / epochs control
    # max_steps takes precedence over num_epochs when > 0
    max_steps: int = -1                    # -1 = derive from num_epochs × dataset

    # Data
    train_data_path: str = ""
    db_dir: str = ""

    # Correction multitask (Stage 2 extension)
    correction_data_path: str = ""          # path to mined correction JSONL
    correction_mix_ratio: float = 0.2       # fraction of correction samples per batch
