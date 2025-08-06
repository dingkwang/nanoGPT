# nanoGPT API Reference

This document provides detailed API documentation for the core components of nanoGPT.

## Table of Contents

- [Model Components](#model-components)
  - [GPTConfig](#gptconfig)
  - [GPT](#gpt)
  - [Block](#block)
  - [CausalSelfAttention](#causalselfattention)
  - [MLP](#mlp)
- [Training Components](#training-components)
  - [Training Script Parameters](#training-script-parameters)
  - [Data Loading](#data-loading)
  - [Optimizer Configuration](#optimizer-configuration)
- [Utilities](#utilities)
  - [Configurator](#configurator)
  - [Benchmarking](#benchmarking)
  - [Sampling](#sampling)

## Model Components

### GPTConfig

Configuration class for GPT model parameters.

```python
@dataclass
class GPTConfig:
    block_size: int = 1024      # Context length
    vocab_size: int = 50304     # Vocabulary size (padded to nearest multiple of 64)
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension
    dropout: float = 0.0        # Dropout probability
    bias: bool = True           # Use bias in linear layers and layer norm
```

**Parameters:**
- `block_size`: Maximum sequence length the model can handle
- `vocab_size`: Size of the vocabulary (usually 50257 for GPT-2 BPE, padded to 50304)
- `n_layer`: Number of transformer blocks in the model
- `n_head`: Number of parallel attention heads (must divide `n_embd` evenly)
- `n_embd`: Embedding dimension and hidden size throughout the model
- `dropout`: Dropout rate applied to embeddings, attention, and MLP layers
- `bias`: Whether to include bias terms in linear layers and layer normalization

**Standard Configurations:**
- GPT-2 Small (124M): `n_layer=12, n_head=12, n_embd=768`
- GPT-2 Medium (350M): `n_layer=24, n_head=16, n_embd=1024`
- GPT-2 Large (774M): `n_layer=36, n_head=20, n_embd=1280`
- GPT-2 XL (1558M): `n_layer=48, n_head=25, n_embd=1600`

### GPT

Main GPT model class implementing a decoder-only transformer.

```python
class GPT(nn.Module):
    def __init__(self, config):
        # Initialize model components
        
    def forward(self, idx, targets=None):
        # Forward pass returning logits and optionally loss
        
    def crop_block_size(self, block_size):
        # Crop position embeddings for smaller context
        
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # Load pretrained GPT-2 weights from OpenAI
```

**Methods:**

#### `__init__(self, config: GPTConfig)`
Initializes the GPT model with the given configuration.

#### `forward(self, idx: torch.Tensor, targets: torch.Tensor = None)`
Forward pass through the model.

**Parameters:**
- `idx`: Input token indices of shape `(batch_size, sequence_length)`
- `targets`: Optional target token indices for loss computation

**Returns:**
- `logits`: Output logits of shape `(batch_size, sequence_length, vocab_size)`
- `loss`: Cross-entropy loss if targets provided, otherwise None

#### `crop_block_size(self, block_size: int)`
Reduces the model's context length by cropping position embeddings.

#### `from_pretrained(cls, model_type: str, override_args: dict = None)`
Creates a GPT model initialized with pretrained GPT-2 weights.

**Parameters:**
- `model_type`: One of `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`
- `override_args`: Optional dictionary to override default config parameters

### Block

Transformer block containing self-attention and MLP layers.

```python
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
```

### CausalSelfAttention

Multi-head causal self-attention mechanism.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # QKV projections, output projection, dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
```

**Features:**
- Supports Flash Attention for PyTorch >= 2.0
- Causal masking to prevent attention to future tokens
- Dropout for regularization
- Efficient batched computation of all attention heads

### MLP

Multi-layer perceptron with GELU activation.

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
```

## Training Components

### Training Script Parameters

The `train.py` script accepts numerous command-line arguments for configuration:

#### I/O Parameters
- `--out_dir`: Output directory for checkpoints (default: 'out')
- `--eval_interval`: Evaluation interval in iterations (default: 2000)
- `--log_interval`: Logging interval in iterations (default: 1)
- `--eval_iters`: Number of iterations for evaluation (default: 200)
- `--eval_only`: Exit after first evaluation (default: False)
- `--always_save_checkpoint`: Save checkpoint after each eval (default: True)
- `--init_from`: Initialization mode ('scratch', 'resume', or 'gpt2*')

#### Data Parameters
- `--dataset`: Dataset name (default: 'openwebtext')
- `--gradient_accumulation_steps`: Steps for gradient accumulation (default: 40)
- `--batch_size`: Micro-batch size (default: 12)
- `--block_size`: Context length (default: 1024)

#### Model Parameters
- `--n_layer`: Number of layers (default: 12)
- `--n_head`: Number of attention heads (default: 12)
- `--n_embd`: Embedding dimension (default: 768)
- `--dropout`: Dropout rate (default: 0.0)
- `--bias`: Use bias in linear layers (default: False)

#### Optimizer Parameters
- `--learning_rate`: Peak learning rate (default: 6e-4)
- `--max_iters`: Maximum training iterations (default: 600000)
- `--weight_decay`: L2 regularization (default: 1e-1)
- `--beta1`: Adam beta1 (default: 0.9)
- `--beta2`: Adam beta2 (default: 0.95)
- `--grad_clip`: Gradient clipping value (default: 1.0)

#### Learning Rate Schedule
- `--decay_lr`: Enable learning rate decay (default: True)
- `--warmup_iters`: Warmup iterations (default: 2000)
- `--lr_decay_iters`: Decay iterations (default: 600000)
- `--min_lr`: Minimum learning rate (default: 6e-5)

#### System Parameters
- `--device`: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
- `--dtype`: Data type ('float32', 'bfloat16', 'float16')
- `--compile`: Use PyTorch 2.0 compilation (default: True)

### Data Loading

Data is expected in binary format:
- `train.bin`: Training data as uint16 array of token indices
- `val.bin`: Validation data as uint16 array of token indices

Data preparation scripts are provided in the `data/` directory for common datasets.

### Optimizer Configuration

Uses AdamW optimizer with:
- Cosine learning rate schedule with warmup
- Weight decay applied only to 2D parameters (weights, not biases/norms)
- Gradient clipping for stability

## Utilities

### Configurator

The `configurator.py` module allows configuration override via:
- Command-line arguments
- Configuration files (Python scripts)
- Environment variables

### Benchmarking

The `bench.py` script provides model benchmarking:
- Measures forward pass time
- Profiles memory usage
- Tests different batch sizes and sequence lengths

### Sampling

The `sample.py` script supports:
- Sampling from pretrained models
- Temperature and top-k sampling
- Custom prompts from files
- Batch generation

**Usage Example:**
```bash
python sample.py \
    --init_from=gpt2-xl \
    --start="The meaning of life is" \
    --num_samples=3 \
    --max_new_tokens=50 \
    --temperature=0.8 \
    --top_k=200
```

## Error Handling and Debugging

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size`, `block_size`, or model size
2. **Slow training**: Enable `--compile=True` and use appropriate `dtype`
3. **NaN gradients**: Check learning rate, use gradient clipping
4. **Poor convergence**: Verify data preprocessing and learning rate schedule

### Distributed Training

For multi-GPU training:
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

For multi-node training, specify master address and node ranks as shown in the main README.

## Performance Optimization

1. **Enable compilation**: Use `--compile=True` for PyTorch 2.0+
2. **Use appropriate dtype**: `bfloat16` or `float16` for modern GPUs
3. **Flash Attention**: Automatically used with PyTorch >= 2.0
4. **Gradient accumulation**: Simulate larger batch sizes efficiently
5. **Data loading**: Use memory-mapped files for large datasets