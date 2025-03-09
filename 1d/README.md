# 1D Autoencoder

This implementation demonstrates a neural network architecture that uses 1-dimensional tensors for its layers. The network processes 2D image data that has been flattened into a 1D sequence, representing the simplest form in our exploration of n-dimensional tensor operations.

## Architecture Overview

The 1D autoencoder consists of two main components:

1. **Encoder**: Transforms the flattened 1D sequence into a compressed latent space
2. **Decoder**: Reconstructs the original sequence from the latent space

### Input Representation

We start with a standard 2D grayscale image and flatten it into a 1D tensor. The resulting tensor has shape `[batch_size, channels, sequence_length]` where:

- `batch_size`: Number of samples (typically 1 in our implementation)
- `channels`: Number of input channels (1 for grayscale)
- `sequence_length`: The flattened image pixels (height × width)

Each layer in the network operates on these 1D tensors, with convolutions applied along the sequence dimension.

### Network Structure

The network has 3 layers in both the encoder and decoder, maintaining architectural consistency with our higher-dimensional implementations:

#### Encoder
- Layer 1: 1D Convolution (1→8 channels) + ReLU
- Layer 2: 1D Convolution (8→4 channels) + ReLU
- Layer 3: 1D Convolution (4→4 channels) + ReLU

#### Decoder
- Layer 1: 1D Transposed Convolution (4→4 channels) + ReLU
- Layer 2: 1D Transposed Convolution (4→8 channels) + ReLU
- Layer 3: 1D Transposed Convolution (8→1 channels) + Sigmoid

## 1D Convolution Implementation

The 1D autoencoder uses PyTorch's built-in 1D convolution operations. These operations process the data as a sequence, where each position in the sequence is connected to nearby positions through the convolution kernel.

## Dimensionality Reduction

Through the encoder, the 1D input is progressively reduced in length:
- Each convolution with stride 2 halves the sequence length
- After 3 layers, the sequence length is reduced by a factor of 8
- The number of channels increases and then decreases, forming a bottleneck

For example, with an input of shape `[1, 1, 1048576]` (a flattened 1024×1024 image), the encoded representation would have shape approximately `[1, 4, 131072]`.

## Training Process

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer with learning rate 0.001
- 500 training epochs
- CSV logging of training progress

## Implementation Note

The 1D autoencoder represents the simplest form in our dimensional exploration. While higher-dimensional models preserve spatial relationships in the data, the 1D model treats the image as a pure sequence, losing the 2D spatial structure. This makes it a good baseline to demonstrate the benefits of preserving dimensional structure in the higher-dimensional models.