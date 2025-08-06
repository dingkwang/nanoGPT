# nanoGPT Development Guide

This guide is for developers who want to modify, extend, or contribute to the nanoGPT codebase.

## Table of Contents

- [Code Structure](#code-structure)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Adding New Features](#adding-new-features)
- [Testing and Validation](#testing-and-validation)
- [Performance Optimization](#performance-optimization)
- [Contributing Guidelines](#contributing-guidelines)

## Code Structure

### Core Files

```
nanoGPT/
├── train.py              # Main training script (~300 lines)
├── model.py              # GPT model implementation (~300 lines)
├── sample.py             # Text generation script
├── bench.py              # Benchmarking utilities
├── configurator.py       # Configuration system
├── config/               # Training configurations
│   ├── train_gpt2.py
│   ├── finetune_shakespeare.py
│   └── ...
├── data/                 # Dataset preparation scripts
│   ├── openwebtext/
│   ├── shakespeare/
│   └── shakespeare_char/
├── assets/               # Documentation images
└── notebooks/            # Analysis notebooks
    ├── gpt_dev.ipynb
    ├── scaling_laws.ipynb
    └── transformer_sizing.ipynb
```

### Design Principles

1. **Simplicity**: Keep code readable and hackable
2. **Minimal dependencies**: Only essential packages
3. **Educational value**: Code should be easy to understand
4. **Performance**: Efficient implementation without sacrificing clarity

## Development Setup

### Environment Setup

```bash
# Clone repository
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy transformers datasets tiktoken wandb tqdm

# Install development dependencies
pip install pytest black isort flake8 jupyter
```

### Development Workflow

```bash
# Format code
black train.py model.py sample.py

# Sort imports
isort train.py model.py sample.py

# Lint code
flake8 train.py model.py sample.py

# Run tests
pytest tests/
```

## Architecture Overview

### Model Architecture (model.py)

```python
GPT
├── transformer
│   ├── wte (token embeddings)
│   ├── wpe (position embeddings)
│   ├── drop (embedding dropout)
│   ├── h (transformer blocks)
│   │   └── Block
│   │       ├── ln_1 (layer norm)
│   │       ├── attn (self-attention)
│   │       ├── ln_2 (layer norm)
│   │       └── mlp (feed-forward)
│   └── ln_f (final layer norm)
└── lm_head (language modeling head)
```

### Key Components

#### 1. CausalSelfAttention
- Multi-head self-attention with causal masking
- Supports Flash Attention (PyTorch >= 2.0)
- Efficient batched computation

#### 2. MLP
- Two-layer feed-forward network
- GELU activation function
- 4x expansion ratio (hidden_size = 4 * n_embd)

#### 3. Block
- Pre-norm architecture (LayerNorm before attention/MLP)
- Residual connections
- Dropout for regularization

### Training Loop (train.py)

```python
# Main training components:
1. Data loading (memory-mapped binary files)
2. Model initialization/loading
3. Optimizer setup (AdamW with cosine LR schedule)
4. Training loop with gradient accumulation
5. Evaluation and checkpointing
6. Distributed training support (DDP)
```

## Adding New Features

### 1. Adding New Model Components

**Example: Adding RoPE (Rotary Position Embedding)**

```python
# In model.py
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Pre-compute rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len):
        # Apply rotary embeddings
        # ... implementation details
        return x

# Modify CausalSelfAttention to use RoPE
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code
        if config.use_rope:
            self.rope = RotaryPositionalEmbedding(config.n_embd // config.n_head)
```

### 2. Adding New Optimizers

**Example: Adding Lion optimizer**

```python
# In train.py
def configure_optimizers(model, weight_decay, learning_rate, betas, device_type, optimizer_type='adamw'):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    elif optimizer_type == 'lion':
        from lion_pytorch import Lion
        optimizer = Lion(optim_groups, lr=learning_rate, betas=betas)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
    return optimizer
```

### 3. Adding New Datasets

**Create data preparation script:**

```python
# data/my_dataset/prepare.py
import os
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset

def prepare_data():
    # Load your dataset
    dataset = load_dataset("your_dataset")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize and save
    train_ids = []
    val_ids = []
    
    for split, ids_list in [('train', train_ids), ('validation', val_ids)]:
        for example in dataset[split]:
            text = example['text']
            tokens = enc.encode(text)
            ids_list.extend(tokens)
    
    # Save as binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_ids.tofile('train.bin')
    val_ids.tofile('val.bin')
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    prepare_data()
```

### 4. Adding New Sampling Strategies

**Example: Top-p (nucleus) sampling**

```python
# In sample.py
def top_p_sampling(logits, p=0.9):
    """Apply top-p (nucleus) sampling to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

# Integrate into sampling loop
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        if top_p is not None:
            logits = top_p_sampling(logits, top_p)
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

## Testing and Validation

### Unit Tests

```python
# tests/test_model.py
import torch
import pytest
from model import GPT, GPTConfig

def test_gpt_forward():
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    
    # Test forward pass
    idx = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx)
    
    assert logits.shape == (2, 32, config.vocab_size)
    assert loss is None
    
    # Test with targets
    targets = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx, targets)
    
    assert loss is not None
    assert loss.item() > 0

def test_model_loading():
    # Test pretrained model loading
    model = GPT.from_pretrained('gpt2')
    assert model.config.vocab_size == 50257
    assert model.config.n_layer == 12
```

### Integration Tests

```python
# tests/test_training.py
def test_training_step():
    """Test a single training step."""
    config = GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Sample data
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    
    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0
    
def test_overfitting_single_batch():
    """Test that model can overfit a single batch."""
    config = GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    
    initial_loss = None
    for i in range(100):
        logits, loss = model(x, y)
        if i == 0:
            initial_loss = loss.item()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1  # Should reduce loss significantly
```

### Performance Benchmarks

```python
# benchmarks/bench_model.py
import time
import torch
from model import GPT, GPTConfig

def benchmark_forward_pass():
    config = GPTConfig()
    model = GPT(config).cuda()
    model.eval()
    
    batch_size = 8
    seq_len = 1024
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            logits, _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            logits, _ = model(x)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    print(f"Average forward pass time: {avg_time:.4f}s")
    print(f"Tokens per second: {tokens_per_second:.0f}")
```

## Performance Optimization

### 1. Memory Optimization

**Gradient Checkpointing:**
```python
# In model.py Block class
def forward(self, x):
    if self.training and self.gradient_checkpointing:
        x = x + torch.utils.checkpoint.checkpoint(self.attn, self.ln_1(x))
        x = x + torch.utils.checkpoint.checkpoint(self.mlp, self.ln_2(x))
    else:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
    return x
```

**Memory-Efficient Attention:**
```python
# Use Flash Attention when available
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0,
        is_causal=True
    )
```

### 2. Compute Optimization

**Kernel Fusion:**
```python
# Use torch.compile for automatic optimization
if compile:
    model = torch.compile(model)
```

**Mixed Precision:**
```python
# Use automatic mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    logits, loss = model(X, Y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Data Loading Optimization

**Memory-Mapped Files:**
```python
# Efficient data loading for large datasets
class DataLoader:
    def __init__(self, data_path, block_size, batch_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
    
    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
        return x, y
```

## Contributing Guidelines

### Code Style

1. **Follow PEP 8** with line length of 88 characters
2. **Use type hints** where appropriate
3. **Add docstrings** for public functions and classes
4. **Keep functions small** and focused
5. **Use descriptive variable names**

### Documentation

1. **Update README** for new features
2. **Add docstrings** with examples
3. **Include unit tests** for new functionality
4. **Update configuration guides** if adding new parameters

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** with proper tests
4. **Run tests**: `pytest tests/`
5. **Format code**: `black . && isort .`
6. **Submit pull request** with clear description

### Backward Compatibility

- **Maintain API compatibility** when possible
- **Deprecate features** before removal
- **Document breaking changes** clearly
- **Provide migration guides** for major changes

### Performance Considerations

- **Profile new features** for performance impact
- **Test on different hardware** configurations
- **Measure memory usage** and training speed
- **Optimize critical paths** without sacrificing readability

### Security

- **Validate user inputs** in configuration
- **Sanitize file paths** in data loading
- **Use secure defaults** for network operations
- **Document security implications** of new features

## Common Development Tasks

### Adding a New Attention Mechanism

1. Create new attention class in `model.py`
2. Add configuration parameters to `GPTConfig`
3. Update `CausalSelfAttention` or create alternative
4. Add unit tests for the new mechanism
5. Benchmark performance vs. standard attention
6. Update documentation

### Implementing a New Optimizer

1. Add optimizer selection logic in `train.py`
2. Update configuration system
3. Test convergence on small dataset
4. Compare with AdamW baseline
5. Document hyperparameter recommendations

### Adding Model Parallelism

1. Identify parallelizable components
2. Implement communication patterns
3. Update training loop for multi-GPU
4. Add configuration for parallel settings
5. Test scaling on multiple nodes

This development guide should help you understand the codebase structure and contribute effectively to nanoGPT. Remember to keep changes simple, well-tested, and properly documented.