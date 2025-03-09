# 2D Autoencoder

This implementation demonstrates a neural network architecture that uses 2-dimensional tensors for its layers. The network processes 2D image data directly, serving as the foundation for our exploration of higher-dimensional tensor operations.

## Architecture Overview

The 2D autoencoder consists of two main components:

1. **Encoder**: Transforms the 2D image into a compressed latent space
2. **Decoder**: Reconstructs the original image from the latent space

### Input Representation

We use standard 2D grayscale images as input. The input tensor has shape `[batch_size, channels, height, width]` where:

- `batch_size`: Number of samples (typically 1 in our implementation)
- `channels`: Number of input channels (1 for grayscale)
- `height` and `width`: Spatial dimensions of the 2D image

Each layer in the network operates on these 2D tensors, with convolutions applied across both spatial dimensions.

### Network Structure

The network has 3 layers in both the encoder and decoder, establishing the pattern that we extend to higher dimensions:

#### Encoder
- Layer 1: 2D Convolution (1→8 channels) + ReLU
- Layer 2: 2D Convolution (8→4 channels) + ReLU
- Layer 3: 2D Convolution (4→4 channels) + ReLU

#### Decoder
- Layer 1: 2D Transposed Convolution (4→4 channels) + ReLU
- Layer 2: 2D Transposed Convolution (4→8 channels) + ReLU
- Layer 3: 2D Transposed Convolution (8→1 channels) + Sigmoid

## 2D Convolution Implementation

The 2D autoencoder uses PyTorch's built-in 2D convolution operations, which serve as the foundation for our understanding of convolutional operations in higher dimensions.

## Dimensionality Reduction

Through the encoder, the 2D input is progressively reduced in spatial dimensions:
- Each convolution with stride 2 halves the size in each dimension
- After 3 layers, the spatial dimensions are reduced by a factor of 8
- The number of channels increases and then decreases, forming a bottleneck

For example, with an input of shape `[1, 1, 1024, 1024]`, the encoded representation would have shape approximately `[1, 4, 128, 128]`.

## Training Process

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer with learning rate 0.001
- 500 training epochs
- CSV logging of training progress

## Implementation Note

The 2D autoencoder establishes the architectural pattern that we extend to higher dimensions. It uses the same channel progression (1→8→4→4→8→1) and activation functions (ReLU between layers, Sigmoid at output) that we maintain across all our n-dimensional implementations.