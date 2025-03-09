# 4D Autoencoder

This implementation demonstrates a neural network architecture that uses 4-dimensional tensors for its layers. The network processes 2D image data that has been artificially expanded into a 4D tensor format by repeating the image across multiple synthetic dimensions. This approach allows us to explore how convolutional operations can be extended to higher-dimensional spaces.

## Architecture Overview

The 4D autoencoder consists of two main components:

1. **Encoder**: Transforms the expanded 4D representation into a compressed latent space
2. **Decoder**: Reconstructs the representation from the latent space

### Input Representation

We start with a standard 2D grayscale image and expand it into a 4D tensor by repeating the image data across multiple synthetic dimensions. The resulting tensor has shape `[batch_size, channels, time, depth, height, width]` where:

- `batch_size`: Number of samples (typically 1 in our implementation)
- `channels`: Number of input channels (1 for grayscale)
- `time`: Fourth dimension
- `depth`: Third dimension
- `height` and `width`: Original spatial dimensions of the 2D image

Each layer in the network operates on these 4D tensors, with convolutions applied across all dimensions.

### Network Structure

The network has 3 layers in both the encoder and decoder, maintaining architectural consistency with our other dimensional implementations:

#### Encoder
- Layer 1: 4D Convolution (1→8 channels) + ReLU
- Layer 2: 4D Convolution (8→4 channels) + ReLU
- Layer 3: 4D Convolution (4→4 channels) + ReLU

#### Decoder
- Layer 1: 4D Transposed Convolution (4→4 channels) + ReLU
- Layer 2: 4D Transposed Convolution (4→8 channels) + ReLU
- Layer 3: 4D Transposed Convolution (8→1 channels) + Sigmoid

## 4D Convolution Implementation

Since PyTorch doesn't natively support 4D convolutions, we implement them using a hierarchical approach:

### Hierarchical Convolution Strategy

The 4D autoencoder uses a single-level hierarchical approach:

1. **Single Hierarchy**: 
   - 4D convolutions use 3D convolutions (PyTorch native)

2. **Dimension Processing Order**:
   - First process the time dimension (T)
   - Then process the spatial dimensions (D, H, W) using native 3D convolutions

3. **Sliding Window Approach**: For the time dimension, we:
   - Take 3 consecutive time slices (t, t+1, t+2)
   - Apply separate 3D convolutions to each slice
   - Average the results to create one output time slice
   - Repeat with stride to reduce the time dimension

### Implementation Details

The `Conv4d` class processes the time dimension by:
1. Taking 3 consecutive time slices (t, t+1, t+2)
2. Applying separate 3D convolutions to each slice
3. Averaging the results to create one output time slice
4. Repeating this process with stride to reduce the time dimension

## Dimensionality Reduction

Through the encoder, the 4D input is progressively reduced in spatial dimensions:
- Each convolution with stride 2 halves the size in each dimension
- After 3 layers, the spatial dimensions are reduced by a factor of 8
- The time dimension is also reduced due to the strided processing
- The number of channels increases and then decreases, forming a bottleneck

## Shape Handling

A key challenge in the 4D autoencoder is handling shape mismatches between the input and output tensors, particularly in the time dimension. The implementation includes:

1. **Training adaptation**: During training, we handle shape mismatches by either:
   - Taking a slice of the input that matches the output dimensions
   - Repeating the output along dimensions to match the input

2. **Visualization adaptation**: For visualization, we ensure proper shape matching by:
   - Repeating the output along the time dimension
   - Padding with the last time slice if needed

## Training Process

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer with learning rate 0.001
- 500 training epochs
- CSV logging of training progress
