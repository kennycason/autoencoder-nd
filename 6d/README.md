# 6D Autoencoder

This implementation demonstrates a neural network architecture that uses 6-dimensional tensors for its layers. The network processes 2D image data that has been artificially expanded into a 6D tensor format by repeating the image across multiple synthetic dimensions. This approach allows us to explore how convolutional operations can be extended to higher-dimensional spaces.

## Architecture Overview

The 6D autoencoder consists of two main components:

1. **Encoder**: Transforms the expanded 6D representation into a compressed latent space
2. **Decoder**: Reconstructs the representation from the latent space

### Input Representation

We start with a standard 2D grayscale image and expand it into a 6D tensor by repeating the image data across multiple synthetic dimensions. The resulting tensor has shape `[batch_size, channels, extra_dim, c_dim, time, depth, height, width]` where:

- `batch_size`: Number of samples (typically 1 in our implementation)
- `channels`: Number of input channels (1 for grayscale)
- `extra_dim`: Sixth dimension 
- `c_dim`: Fifth dimension
- `time`: Fourth dimension
- `depth`: Third dimension
- `height` and `width`: Original spatial dimensions of the 2D image

Each layer in the network operates on these 6D tensors, with convolutions applied across all dimensions.

### Network Structure

The network has 3 layers in both the encoder and decoder, maintaining architectural consistency with our lower-dimensional implementations:

#### Encoder
- Layer 1: 6D Convolution (1→8 channels) + ReLU
- Layer 2: 6D Convolution (8→4 channels) + ReLU
- Layer 3: 6D Convolution (4→4 channels) + ReLU

#### Decoder
- Layer 1: 6D Transposed Convolution (4→4 channels) + ReLU
- Layer 2: 6D Transposed Convolution (4→8 channels) + ReLU
- Layer 3: 6D Transposed Convolution (8→1 channels) + Sigmoid

## 6D Convolution Implementation

Since PyTorch doesn't natively support 6D convolutions, we implement them using a triple hierarchical approach:

### Triple Hierarchical Convolution Strategy

The 6D autoencoder represents the pinnacle of our hierarchical approach:

1. **Triple Hierarchy**: 
   - 6D convolutions use 5D convolutions
   - 5D convolutions use 4D convolutions
   - 4D convolutions use 3D convolutions (PyTorch native)

2. **Dimension Processing Order**:
   - First process the extra dimension (E)
   - Then process the channel dimension (C)
   - Then process the time dimension (T)
   - Finally process the spatial dimensions (D, H, W) using native 3D convolutions

3. **Sliding Window Approach**: For each higher dimension, we:
   - Take 3 consecutive slices (e.g., e, e+1, e+2)
   - Apply separate convolutions to each slice
   - Average the results to create one output slice
   - Repeat with stride to reduce the dimension

4. **Nested Processing**: Each level of the hierarchy follows the same pattern, creating a deeply nested structure that processes all six dimensions.

### Implementation Details

The `Conv6d` class processes the extra dimension by:
