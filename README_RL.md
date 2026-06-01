# Dr. CPPO Reinforcement Learning Training for Cadrille

This module implements **Dr. CPPO** (Derivative-free GRPO + CPPO) for online reinforcement learning fine-tuning of Cadrille on point cloud / mesh inputs.

## Overview

Dr. CPPO combines two recent modifications for efficient RL training:

1. **Dr. GRPO** (Liu et al. 2025): Eliminates the need for a separate reference model, reducing memory overhead
2. **CPPO** (Lin et al. 2025): Uses samples with the strongest advantage signal, focusing gradient updates on informative trajectories

### Key Features

- **Reward Functions**: IoU (Intersection over Union) or Chamfer distance metrics
- **Point Cloud Focus**: Designed for point cloud / mesh inputs with proper preprocessing
- **Memory Efficient**: Single model training (no reference model during generation)
- **Safe Execution**: CadQuery code generation runs in isolated subprocesses to prevent memory leaks

## Installation

```bash
pip install -r rl_requirements.txt
```

## Quick Start

### Basic RL Training on Point Clouds

```bash
python rl_trainer.py \
    --data-path ./data \
    --output-dir ./rl_checkpoints \
    --mode pc \
    --num-steps 5000
```

### With Text Modality

```bash
python rl_trainer.py \
    --data-path ./data \
    --output-dir ./rl_checkpoints \
    --mode pc \
    --use-text \
    --num-steps 10000
```

### With Image Modality

```bash
python rl_trainer.py \
    --data-path ./data \
    --output-dir ./rl_checkpoints \
    --mode pc_img \
    --num-steps 10000
```

## Configuration

Edit `DrCPPOConfig` in `rl_trainer.py`:

```python
config = DrCPPOConfig(
    # Rollout collection
    num_rollouts_per_batch=4,          # G: number of sequences to sample per input
    num_samples_for_update=2,          # N: top samples for policy update
    ppo_clip_ratio=0.2,                # epsilon: PPO clipping range
    ppo_epochs=2,                      # PPO optimization epochs
    
    # Reward
    reward_metric="iou",               # "iou" or "chamfer"
    chamfer_n_points=2048,             # points for Chamfer distance
    
    # Training
    learning_rate=5e-5,
    temperature=1.0,                   # sampling temperature
    max_gen_length=512,                # max tokens to generate
    
    # Sampling
    top_p=0.9,
    top_k=50,
)
```

## Algorithm Details

The Dr. CPPO training loop for each batch:

1. **Sampling**: Generate G sequences from current policy π_θ with temperature T=1.0
2. **Reward Computation**: Evaluate each sequence using IoU or Chamfer distance
3. **Advantage Estimation**: A_g = r_g - mean({r_i})
4. **Sample Selection**: Select N samples with highest |A_g|
5. **PPO Update**: Optimize policy with clipped objective:

```
L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
```

## Reward Functions

### IoU Reward
- Computes intersection over union between generated and ground truth meshes
- Range: [0, 1]
- Stronger signal for geometric accuracy

### Chamfer Distance Reward
- Symmetric distance between surface points
- Normalized as: reward = exp(-distance)
- Continuous gradient signal

## File Structure

```
cadrille/
├── rl_trainer.py           # Main Dr. CPPO trainer
├── rl_requirements.txt     # RL-specific dependencies
├── README_RL.md           # This file
├── cadrille.py            # Model definition
├── dataset.py             # Dataset loaders
├── train.py               # SFT training script
├── evaluate.py            # Evaluation utilities
└── test.py                # Inference script
```

## Checkpointing

Models are saved every `eval_every` steps (default: 1000):

```
./rl_checkpoints/
├── checkpoint_step_0/
│   ├── model/                (HuggingFace model format)
│   ├── processor/            (Tokenizer + processor)
│   └── optimizer.pt
├── checkpoint_step_1000/
│   └── ...
└── ...
```

Resume training from checkpoint:

```python
trainer = DrCPPOTrainer(...)
trainer.model = Cadrille.from_pretrained("./rl_checkpoints/checkpoint_step_5000/model")
trainer.train(dataloader, num_steps=10000)
```

## Monitoring

Training logs include:

- `ppo_loss`: Policy gradient loss
- `mean_reward`: Average reward across rollouts
- `max_reward`: Best reward in batch
- `min_reward`: Worst reward in batch
- `mean_advantage`: Average advantage estimate

## Common Issues

### Memory Issues
- Reduce `num_rollouts_per_batch` or `batch_size`
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()`

### Slow Reward Computation
- Reduce `chamfer_n_points` for faster sampling
- Use IoU instead of Chamfer (faster but lower signal quality)
- Use multiprocessing pool for parallel evaluation

### CadQuery Crashes
- Subprocess isolation is enabled by default
- Increase timeout in `RewardComputer.py_string_to_mesh()` if needed

## Citation

If you use this RL trainer, please cite the original Cadrille paper:

```bibtex
@article{kolodiazhnyi2025cadrille,
  title={cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning},
  author={Kolodiazhnyi, Maksim and Tarasov, Denis and Zhemchuzhnikov, Dmitrii and Nikulin, Alexander and Zisman, Ilya and Vorontsova, Anna and Konushin, Anton and Kurenkov, Vladislav and Rukhovich, Danila},
  journal={arXiv preprint arXiv:2505.22914},
  year={2025}
}
```

## References

- **Dr. GRPO**: Liu et al. (2025) - Derivative-free GRPO
- **CPPO**: Lin et al. (2025) - Contrastive PPO with preference samples
- **Original Cadrille**: https://arxiv.org/abs/2505.22914

## Future Improvements

- [ ] Multi-GPU training with FSDP
- [ ] Reward normalization/scaling strategies
- [ ] Curriculum learning (start with simple shapes)
- [ ] Batch RL with offline data
- [ ] Intrinsic exploration bonuses
