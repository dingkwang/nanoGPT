# nanoGPT Configuration Guide

This guide provides detailed information about configuring nanoGPT for different use cases, hardware setups, and training objectives.

## Table of Contents

- [Configuration System](#configuration-system)
- [Model Scaling](#model-scaling)
- [Training Configuration](#training-configuration)
- [Hardware-Specific Settings](#hardware-specific-settings)
- [Dataset Configuration](#dataset-configuration)
- [Optimization Settings](#optimization-settings)
- [Example Configurations](#example-configurations)

## Configuration System

nanoGPT uses a flexible configuration system that supports multiple override methods:

### 1. Command Line Arguments
```bash
python train.py --batch_size=32 --learning_rate=3e-4 --max_iters=100000
```

### 2. Configuration Files
Create Python files in the `config/` directory:
```python
# config/my_experiment.py
batch_size = 64
learning_rate = 1e-3
max_iters = 50000
out_dir = 'out-my-experiment'
```

Use with: `python train.py config/my_experiment.py`

### 3. Environment Variables
```bash
export batch_size=32
export learning_rate=3e-4
python train.py
```

### Priority Order
1. Command line arguments (highest priority)
2. Configuration file parameters
3. Environment variables
4. Default values (lowest priority)

## Model Scaling

### Size Configurations

| Model Size | Parameters | n_layer | n_head | n_embd | Memory (training) |
|------------|------------|---------|--------|--------|------------------|
| Tiny       | ~10M       | 4       | 4      | 128    | ~1GB            |
| Small      | ~40M       | 6       | 6      | 384    | ~2GB            |
| Medium     | ~124M      | 12      | 12     | 768    | ~4GB            |
| Large      | ~350M      | 24      | 16     | 1024   | ~8GB            |
| XL         | ~774M      | 36      | 20     | 1280   | ~16GB           |
| XXL        | ~1.5B      | 48      | 25     | 1600   | ~32GB           |

### Scaling Rules

**General principles:**
- `n_embd` should be divisible by `n_head`
- Larger models generally need lower learning rates
- Context length (`block_size`) affects memory quadratically
- Use gradient accumulation for effective large batch sizes

**Parameter relationships:**
```python
# Approximate parameter count
params = n_layer * (12 * n_embd**2) + vocab_size * n_embd
```

## Training Configuration

### Learning Rate Scheduling

**Cosine decay with warmup (recommended):**
```python
decay_lr = True
warmup_iters = 2000          # Warmup period
lr_decay_iters = max_iters   # Decay over full training
min_lr = learning_rate / 10  # Final LR (10% of peak)
```

**Constant learning rate:**
```python
decay_lr = False
learning_rate = 6e-4  # Fixed throughout training
```

**Custom schedule:**
```python
# Modify get_lr() function in train.py
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    elif it < plateau_iters:
        return learning_rate
    else:
        return learning_rate * 0.1
```

### Batch Size Strategy

**Effective batch size calculation:**
```
effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
```

**Recommendations:**
- Start with `batch_size=12` for 40GB GPUs
- Use `gradient_accumulation_steps` to reach target effective batch size
- Common effective batch sizes: 64-512 for small models, 32-128 for large models

**Memory-efficient training:**
```python
batch_size = 8                    # Smaller micro-batches
gradient_accumulation_steps = 16  # More accumulation steps
# Effective batch size = 8 * 16 = 128
```

### Regularization

**Dropout (not recommended for large models):**
```python
dropout = 0.0  # Modern large models work better without dropout
```

**Weight decay:**
```python
weight_decay = 1e-1  # Standard value, applied only to 2D parameters
```

**Gradient clipping:**
```python
grad_clip = 1.0  # Prevents gradient explosion
```

## Hardware-Specific Settings

### Single GPU (Consumer Cards)

**RTX 3090/4090 (24GB):**
```python
# Small model
n_layer = 12
n_head = 12
n_embd = 768
batch_size = 8
gradient_accumulation_steps = 8
block_size = 1024
dtype = 'bfloat16'  # If supported, else 'float16'
compile = True
```

**RTX 3080 (10GB):**
```python
# Tiny model
n_layer = 6
n_head = 6
n_embd = 384
batch_size = 4
gradient_accumulation_steps = 16
block_size = 512
dtype = 'float16'
compile = True
```

### Multi-GPU (Data Center)

**8x A100 40GB:**
```python
# GPT-2 reproduction
n_layer = 12
n_head = 12
n_embd = 768
batch_size = 12
gradient_accumulation_steps = 5
block_size = 1024
dtype = 'bfloat16'
compile = True

# Run with: torchrun --standalone --nproc_per_node=8 train.py
```

**8x H100 80GB:**
```python
# Large model
n_layer = 24
n_head = 16
n_embd = 1024
batch_size = 16
gradient_accumulation_steps = 2
block_size = 2048
dtype = 'bfloat16'
compile = True
```

### CPU Training

**For development/testing:**
```python
device = 'cpu'
compile = False
dtype = 'float32'
batch_size = 4
gradient_accumulation_steps = 1
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
max_iters = 1000
eval_interval = 100
```

### Apple Silicon (MPS)

```python
device = 'mps'
compile = False  # Not supported on MPS yet
dtype = 'float32'
batch_size = 8
gradient_accumulation_steps = 4
block_size = 512
```

## Dataset Configuration

### Built-in Datasets

**OpenWebText (large scale):**
```python
dataset = 'openwebtext'
# Requires ~54GB preprocessed data
# Run: python data/openwebtext/prepare.py
```

**Shakespeare (character-level):**
```python
dataset = 'shakespeare_char'
# Small dataset for quick experiments
# Run: python data/shakespeare_char/prepare.py
```

**Shakespeare (token-level):**
```python
dataset = 'shakespeare'
# Uses GPT-2 BPE tokenization
# Run: python data/shakespeare/prepare.py
```

### Custom Datasets

**Directory structure:**
```
data/my_dataset/
├── prepare.py          # Data preprocessing script
├── train.bin           # Training data (uint16 array)
├── val.bin            # Validation data (uint16 array)
└── meta.pkl           # Metadata (vocab_size, etc.)
```

**prepare.py template:**
```python
import os
import pickle
import requests
import numpy as np
import tiktoken

# Download and preprocess your data
# Save as train.bin and val.bin (uint16 arrays)
# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,  # index to string mapping
    'stoi': stoi,  # string to index mapping
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
```

## Optimization Settings

### AdamW Parameters

**Standard settings:**
```python
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
```

**Fine-tuning settings:**
```python
learning_rate = 3e-5  # Lower for fine-tuning
weight_decay = 1e-2   # Lighter regularization
beta1 = 0.9
beta2 = 0.999         # Higher beta2 for stability
```

### Memory Optimization

**Enable gradient checkpointing:**
```python
# Modify model.py to add:
# self.gradient_checkpointing = True
# Use torch.utils.checkpoint in forward pass
```

**Mixed precision training:**
```python
dtype = 'bfloat16'  # Preferred for modern GPUs
# or
dtype = 'float16'   # Better compatibility
```

**Reduce memory usage:**
```python
batch_size = 1                    # Minimal micro-batch
gradient_accumulation_steps = 64  # Large accumulation
eval_iters = 20                   # Fewer eval iterations
always_save_checkpoint = False    # Skip some checkpoints
```

## Example Configurations

### Development/Debugging
```python
# config/debug.py
# Fast training for code testing
out_dir = 'out-debug'
eval_interval = 10
log_interval = 1
eval_iters = 5
max_iters = 100
batch_size = 2
block_size = 64
n_layer = 2
n_head = 2
n_embd = 32
compile = False
```

### Quick Experiment
```python
# config/quick_experiment.py
# Reasonable training in ~1 hour
out_dir = 'out-quick'
dataset = 'shakespeare_char'
eval_interval = 250
max_iters = 5000
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
learning_rate = 1e-3
```

### Production Training
```python
# config/production.py
# Full-scale training
out_dir = 'out-production'
dataset = 'openwebtext'
wandb_log = True
wandb_project = 'gpt-production'
eval_interval = 2000
max_iters = 600000
batch_size = 12
gradient_accumulation_steps = 40
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
learning_rate = 6e-4
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
```

### Fine-tuning
```python
# config/finetune_custom.py
# Fine-tune from GPT-2 checkpoint
init_from = 'gpt2'  # or 'gpt2-medium', etc.
out_dir = 'out-finetune'
dataset = 'my_domain_data'
eval_interval = 500
max_iters = 10000
batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 3e-5
decay_lr = True
warmup_iters = 100
lr_decay_iters = 10000
min_lr = 1e-6
```

## Monitoring and Logging

### Weights & Biases Integration
```python
wandb_log = True
wandb_project = 'my-gpt-project'
wandb_run_name = 'experiment-1'
```

### Local Logging
```python
log_interval = 1        # Log every iteration
eval_interval = 1000    # Evaluate every 1000 iterations
eval_iters = 200        # Use 200 iterations for evaluation
```

### Checkpointing
```python
always_save_checkpoint = True   # Save after every evaluation
# Checkpoints saved to {out_dir}/ckpt.pt
```

## Troubleshooting Common Issues

### Out of Memory
1. Reduce `batch_size`
2. Reduce `block_size`
3. Increase `gradient_accumulation_steps`
4. Use `dtype='float16'`
5. Reduce model size (`n_layer`, `n_embd`)

### Slow Training
1. Enable `compile=True`
2. Use appropriate `dtype` (`bfloat16` > `float16` > `float32`)
3. Ensure Flash Attention is available (PyTorch >= 2.0)
4. Check data loading efficiency

### Poor Convergence
1. Check learning rate (too high/low)
2. Verify data preprocessing
3. Ensure sufficient training time
4. Monitor gradient norms
5. Check for data quality issues

### Numerical Instability
1. Use `grad_clip=1.0`
2. Reduce learning rate
3. Use `dtype='float32'` for debugging
4. Check for corrupted data