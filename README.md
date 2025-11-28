# Building a GPT Model Using the Tiny Shakespeare Dataset

This project walks through the process of training a character-level GPT style model on the Tiny Shakespeare dataset. The goal is to understand how language models work from the inside by building one step by step with PyTorch.

## Overview

The project starts by downloading and loading the Tiny Shakespeare text. Once the dataset is in place, the text is processed to extract the vocabulary, map characters to integers, and convert the entire dataset into tensors. The data is then split into training and validation sets.

From there, a simple bigram model is implemented to introduce the basic idea of predicting the next token from the current one. After that, the project moves toward a more complete Transformer architecture including self attention heads, multi head attention, feed forward layers, positional embeddings, and layer normalization.

The training loop uses batches of sequences taken from the dataset. The model learns to predict the next character by minimizing cross entropy loss. Once training is complete, the model can generate sequences of text in the Shakespeare style.

## Key Features

### Data Preparation

The text dataset is read from a file and all unique characters are identified. Each character is assigned an integer index and the full text is encoded as a tensor of indices. The data is split into training and validation portions.

### Batch Generation

A helper function creates random batches of sequences for both training and validation splits. Each batch contains input sequences and the corresponding target sequences that the model must predict.

### Bigram Model

A simple model is introduced where each character directly predicts the next one based on an embedding lookup table. This step helps visualize the structure of inputs and targets as well as the loss calculation.

### Transformer Components

The project builds several important pieces of a Transformer, including the self attention mechanism, multiple attention heads working in parallel, feed forward layers, and residual connections combined with layer normalization.

### Full GPT Style Model

A complete model is assembled that combines token embeddings, positional embeddings, stacked Transformer blocks, and a final linear layer projecting to vocabulary size. The model can be trained on the dataset and used to generate new text.

### Training Loop

The model is trained using AdamW optimization. The script prints training and validation losses periodically. Once finished, the model can generate new characters based on the learned probability distribution.

## How to Run

1. Download the Tiny Shakespeare text file using the provided command.
2. Place the file in the same directory as the script.
3. Run the script in an environment with PyTorch installed.
4. The training process will begin and, once complete, the model will print a sample of generated text.

