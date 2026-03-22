# Implementing a Decoder-Only GPT Model

This repository contains a custom implementation of a **decoder-only GPT** model trained on the Tiny Shakespeare dataset, using the **MLX** library.

## Model Architecture

The model is a **decoder-only GPT** based on the Transformer decoder from [Attention Is All You Need](https://arxiv.org/pdf/1706.03762).
- **Embedding & Positional Encoding:** Standard token embeddings with sinusoidal positional encodings.
- **Transformer Blocks:** 6 layers of decoder blocks, each with:
  - Multi-Head Self-Attention (6 heads, dimension 384) with causal masking
  - Feed-Forward Network (hidden dimension 4× model dimension)
  - Layer Normalization and residual connections
- **Output:** Linear projection to the vocabulary size for next-token prediction.

## Training Settings

- **Sequence length:** 256
- **Batch size:** 32
- **Dropout:** 0.1 throughout the model
- **Loss function:** Cross-Entropy
- **Optimizer:** AdamW with weight decay = 0.1
- **Learning rate schedule:** hybrid schedule similar to OneCycle:
  - Linear warmup for 10% of total steps
  - Cosine decay to final LR for remaining steps
  - **Max LR:** 3e-4
- **Total steps:** 10,000
- The **best model is saved based on the lowest validation loss** during training.

## File Structure

- [`modules/`](modules/) – core modules for the project:
  - [`dataloader.py`](modules/dataloader.py) – is responsible for creating training batches
  - [`model.py`](modules/model.py) – contains all GPT building blocks and the full model implementation
  - [`tokenizer.py`](modules/tokenizer.py) – implements a character-level tokenizer with encoding and decoding
- [`analysis.ipynb`](analysis.ipynb) – notebook with training and inference analysis
- [`train.py`](train.py) – full training script

## Author

Created by [Denys Bondarchuk](https://github.com/thejvdev). Feel free to reach out or contribute to the project!
