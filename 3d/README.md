# 3D Autoencoder

This implementation demonstrates a neural network architecture that uses 3-dimensional tensors for its layers. The network processes 2D image data that has been artificially expanded into a 3D tensor format by repeating the image across a synthetic depth dimension. This approach allows us to explore how convolutional operations can be extended to higher-dimensional spaces.

## Architecture Overview

The 3D autoencoder consists of two main components:

1. **Encoder**: Transforms the expanded 3D representation into a compressed latent space
2. **Decoder**: Reconstructs the representation from the latent space

### Input Representation

We start with a standard 2D grayscale image and expand it into a 3D tensor by repeating the image data across a synthetic depth dimension. The resulting tensor has shape `[batch_size, channels, depth, height, width]` where:

- `batch_size`: Number of samples (typically 1 in our implementation)
- `channels`: Number of input channels (1 for grayscale)
- `depth`: Third dimension
- `height` and `width`: Original spatial dimensions of the 2D image

Each layer in the network operates on these 3D tensors, with convolutions applied across all dimensions.

### Network Structure

The network has 3 layers in both the encoder and decoder, maintaining architectural consistency with our other dimensional implementations:

#### Encoder
- Layer 1: 3D Convolution (1→8 channels) + ReLU
- Layer 2: 3D Convolution (8→4 channels) + ReLU
- Layer 3: 3D Convolution (4→4 channels) + ReLU

#### Decoder
- Layer 1: 3D Transposed Convolution (4→4 channels) + ReLU
- Layer 2: 3D Transposed Convolution (4→8 channels) + ReLU
- Layer 3: 3D Transposed Convolution (8→1 channels) + Sigmoid

## 3D Convolution Implementation

Unlike our higher-dimensional implementations, the 3D autoencoder can use PyTorch's built-in 3D convolution operations. This provides a foundation for our hierarchical approach to higher dimensions.

## Dimensionality Reduction

Through the encoder, the 3D input is progressively reduced in spatial dimensions:
- Each convolution with stride 2 halves the size in each dimension
- After 3 layers, the spatial dimensions are reduced by a factor of 8
- The number of channels increases and then decreases, forming a bottleneck

For example, with an input of shape `[1, 1, 8, 1024, 1024]`, the encoded representation would have shape approximately `[1, 4, 1, 128, 128]`.

## Training Process

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer with learning rate 0.001
- 500 training epochs
- CSV logging of training progress

## Implementation Note

The 3D autoencoder uses PyTorch's native 3D convolutions, which serve as the foundation for our hierarchical implementations in higher dimensions. By understanding how 3D convolutions work on volumetric data, we can extend the same principles to 4D, 5D, and 6D through our custom hierarchical approach.