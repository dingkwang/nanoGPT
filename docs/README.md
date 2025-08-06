# nanoGPT Documentation

Welcome to the comprehensive documentation for nanoGPT - a minimal, fast, and educational implementation of GPT-2 training and inference.

## 📖 Documentation Overview

This documentation collection supplements the main [README.md](../README.md) with detailed guides for different use cases and user types.

### Quick Navigation

| Document | Description | Target Audience |
|----------|-------------|-----------------|
| [**Main README**](../README.md) | Project overview, quick start guide, and basic usage | All users |
| [**API Reference**](API_REFERENCE.md) | Detailed API documentation for all classes and functions | Developers & advanced users |
| [**Configuration Guide**](CONFIGURATION_GUIDE.md) | Complete guide to configuring training parameters | ML practitioners |
| [**Development Guide**](DEVELOPMENT_GUIDE.md) | Guide for extending and contributing to the codebase | Contributors & researchers |

## 🚀 Getting Started

1. **New to nanoGPT?** Start with the [main README](../README.md)
2. **Want to train a model?** Check the [Configuration Guide](CONFIGURATION_GUIDE.md)
3. **Need API details?** Refer to the [API Reference](API_REFERENCE.md)
4. **Planning to contribute?** Read the [Development Guide](DEVELOPMENT_GUIDE.md)

## 📚 Documentation Structure

### [Main README](../README.md)
The primary entry point covering:
- Installation and dependencies
- Quick start examples (Shakespeare, GPT-2 reproduction)
- Basic usage patterns
- Hardware requirements
- Community resources

### [API Reference](API_REFERENCE.md)
Comprehensive API documentation including:
- Model components (`GPT`, `GPTConfig`, `Block`, etc.)
- Training script parameters
- Data loading and preparation
- Sampling and inference utilities
- Error handling and debugging

### [Configuration Guide](CONFIGURATION_GUIDE.md)
Detailed configuration documentation covering:
- Configuration system overview
- Model scaling guidelines
- Hardware-specific settings
- Dataset configuration
- Optimization parameters
- Example configurations for common scenarios

### [Development Guide](DEVELOPMENT_GUIDE.md)
Developer-focused documentation including:
- Codebase architecture overview
- Development setup and workflow
- Adding new features and components
- Testing and validation strategies
- Performance optimization techniques
- Contributing guidelines

## 🔧 Key Features Covered

### Training & Fine-tuning
- **From scratch training**: Complete GPT-2 reproduction
- **Fine-tuning**: Adapt pretrained models to custom datasets
- **Multi-GPU training**: Distributed training with PyTorch DDP
- **Memory optimization**: Gradient accumulation, mixed precision

### Model Architecture
- **Transformer implementation**: Clean, readable GPT architecture
- **Flash Attention**: Optimized attention for PyTorch 2.0+
- **Configurable models**: From tiny (10M) to large (1.5B+) parameters
- **Pretrained compatibility**: Load OpenAI GPT-2 checkpoints

### Data & Preprocessing
- **Built-in datasets**: OpenWebText, Shakespeare (char/token level)
- **Custom datasets**: Guidelines for preparing your own data
- **Efficient loading**: Memory-mapped binary data format
- **Tokenization**: GPT-2 BPE tokenizer integration

### Inference & Sampling
- **Multiple strategies**: Temperature, top-k, top-p sampling
- **Batch generation**: Efficient multi-sample generation
- **Custom prompts**: File-based prompt loading
- **Model checkpoints**: Easy loading of trained models

## 🛠️ Common Use Cases

### Research & Experimentation
- Quick prototyping with character-level models
- Scaling law experiments and analysis
- Architecture modifications and ablations
- Custom dataset experiments

### Production & Deployment
- Fine-tuning for domain-specific applications
- Efficient inference setup
- Model optimization and compression
- Multi-GPU training orchestration

### Education & Learning
- Understanding transformer architecture
- Hands-on language modeling experience
- Training dynamics visualization
- Code exploration and modification

## 📋 Example Workflows

### 1. Quick Experiment
```bash
# Train character-level model on Shakespeare
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

### 2. GPT-2 Reproduction
```bash
# Prepare OpenWebText dataset
python data/openwebtext/prepare.py
# Train with 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

### 3. Fine-tuning
```bash
# Fine-tune GPT-2 on custom dataset
python train.py config/finetune_shakespeare.py
python sample.py --out_dir=out-shakespeare
```

## 🔍 Finding Information

### By Topic
- **Installation**: [Main README](../README.md#install)
- **Training**: [Configuration Guide](CONFIGURATION_GUIDE.md#training-configuration)
- **Models**: [API Reference](API_REFERENCE.md#model-components)
- **Performance**: [Development Guide](DEVELOPMENT_GUIDE.md#performance-optimization)

### By User Type
- **Beginners**: Start with [Main README](../README.md)
- **ML Practitioners**: Focus on [Configuration Guide](CONFIGURATION_GUIDE.md)
- **Developers**: Use [API Reference](API_REFERENCE.md)
- **Contributors**: Read [Development Guide](DEVELOPMENT_GUIDE.md)

### By Hardware
- **Single GPU**: [Configuration Guide - Single GPU](CONFIGURATION_GUIDE.md#single-gpu-consumer-cards)
- **Multi-GPU**: [Configuration Guide - Multi-GPU](CONFIGURATION_GUIDE.md#multi-gpu-data-center)
- **CPU Only**: [Configuration Guide - CPU Training](CONFIGURATION_GUIDE.md#cpu-training)
- **Apple Silicon**: [Configuration Guide - Apple Silicon](CONFIGURATION_GUIDE.md#apple-silicon-mps)

## 🤝 Contributing to Documentation

We welcome contributions to improve this documentation! Here's how:

1. **Report Issues**: Found unclear explanations or missing information? [Open an issue](https://github.com/karpathy/nanoGPT/issues)
2. **Suggest Improvements**: Have ideas for better organization or new content? Let us know!
3. **Submit Pull Requests**: Follow the [Development Guide](DEVELOPMENT_GUIDE.md#contributing-guidelines)

### Documentation Standards
- **Clarity**: Write for your target audience
- **Examples**: Include practical code examples
- **Accuracy**: Test all code snippets
- **Consistency**: Follow existing formatting and style

## 📞 Getting Help

### Community Resources
- **Discord**: Join the [#nanoGPT channel](https://discord.gg/3zy8kqD9Cp)
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community support

### Before Asking for Help
1. Check the relevant documentation section
2. Search existing GitHub issues
3. Try the troubleshooting guides in each document
4. Prepare a minimal reproducible example

## 🔄 Documentation Updates

This documentation is actively maintained and updated with:
- New feature additions
- Community feedback
- Bug fixes and clarifications
- Performance improvements
- Best practice updates

**Last Updated**: When new features or significant changes are added to nanoGPT

---

**Ready to get started?** Choose your path:
- 🏃‍♂️ **Quick start**: [Main README](../README.md#quick-start)
- 🎯 **Train a model**: [Configuration Guide](CONFIGURATION_GUIDE.md#example-configurations)
- 🔧 **Customize code**: [Development Guide](DEVELOPMENT_GUIDE.md#adding-new-features)
- 📖 **Learn the API**: [API Reference](API_REFERENCE.md#model-components)