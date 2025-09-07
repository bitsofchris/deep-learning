# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a deep learning experiments monorepo focused on building time series Transformer models from the ground up. The repository contains educational projects progressing from basic neural networks to advanced time series models, with a focus on learning fundamentals.

## Project Structure

Each experiment is contained in its own subfolder under `code/`:
- `code/00_neural-network-xor/` - Basic neural network from scratch (XOR problem)
- `code/01_nn-mnist-image-classification/` - MNIST classification with PyTorch
- `code/02_stock-intraday-classification/` - Stock price direction prediction
- `code/03_cnn-mnist-image-classification/` - Convolutional neural networks
- `code/05_data-pruning-mnist-image-classification/` - Data pruning experiments
- `code/06_obsidian-rag-fine-tuning/` - RAG with personal knowledge base
- `code/10_fourier_transforms/` - Fourier analysis on time series data
- `code/11_pytorch-time-series/` - PyTorch time series data loading
- `code/12_time-series-fusion-model/` - Advanced time series fusion models

## Common Development Commands

### Activate Virtual env
Before running any Python.

Activate the virtual environment: /Users/chris/repos/deep-learning/.venv

### Code Formatting and Linting
```bash
# Format code with black
black .

# Check code style with flake8
flake8 .
```

### Running Experiments
```bash
# Run individual Python scripts
python code/00_neural-network-xor/nn.py
python code/01_nn-mnist-image-classification/nn-pytorch.py

# Run experiments with configuration files
python code/05_data-pruning-mnist-image-classification/cnn-lenet-5.py --experiments-file experiments_small.yaml
```

### Jupyter Notebooks
```bash
# Start Jupyter lab for interactive development
jupyter lab

# Run specific notebooks (many experiments use .ipynb files)
jupyter notebook code/04_llms-from-scratch-book/3-attention.ipynb
```

### Use proper logging
Use python logging to show output using appropriate levels and show it to STDOUT so I can see progress as you run scripts.

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Data
Put any data files in a /data subfolder for the current project. These are git ignored.
Put any results / images into a /results subfolder for the current project.


### Teaching
This repo is experiments for me to try things and learn. Be sure to include explanations of things we have done, trade offs, reasons for the paths we took, and how things relate to becoming a better time series AI research engineer and AI researcher.


## Architecture Notes

- Each project folder contains its own README.md with specific instructions
- Data is typically loaded locally or generated synthetically
- Models progress from scratch implementations to PyTorch-based solutions
- Experiments often include both .py scripts and .ipynb notebooks
- The `daily-blog/` folder contains learning reflections and progress notes
- The `notes/` folder contains structured learning materials and paper summaries

## Development Workflow

1. Each experiment is self-contained in its subfolder
2. Use Jupyter notebooks for exploration and visualization
3. Convert to Python scripts for final implementations
4. Format code with black and check with flake8 before committing
5. Document learnings in daily-blog entries