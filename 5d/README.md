# 5D Autoencoder

This implementation demonstrates a neural network architecture that uses 5-dimensional tensors for its layers. The network processes 2D image data that has been artificially expanded into a 5D tensor format by repeating the image across multiple synthetic dimensions. This approach allows us to explore how convolutional operations can be extended to higher-dimensional spaces.

## Architecture Overview

The 5D autoencoder consists of two main components:

1. **Encoder**: Transforms the expanded 5D representation into a compressed latent space
2. **Decoder**: Reconstructs the representation from the latent space

### Input Representation

We start with a standard 2D grayscale image and expand it into a 5D tensor by repeating the image data across multiple synthetic dimensions. The resulting tensor has shape `[batch_size, channels, c_dim, time, depth, height, width]` where:

- `batch_size`: Number of samples (typically 1 in our implementation)
- `channels`: Number of input channels (1 for grayscale)
- `c_dim`: Fifth dimension
- `time`: Fourth dimension
- `depth`: Third dimension
- `height` and `width`: Original spatial dimensions of the 2D image

Each layer in the network operates on these 5D tensors, with convolutions applied across all dimensions.

### Network Structure

The network has 3 layers in both the encoder and decoder, maintaining architectural consistency with our other dimensional implementations:

#### Encoder
- Layer 1: 5D Convolution (1→8 channels) + ReLU
- Layer 2: 5D Convolution (8→4 channels) + ReLU
- Layer 3: 5D Convolution (4→4 channels) + ReLU

#### Decoder
- Layer 1: 5D Transposed Convolution (4→4 channels) + ReLU
- Layer 2: 5D Transposed Convolution (4→8 channels) + ReLU
- Layer 3: 5D Transposed Convolution (8→1 channels) + Sigmoid

## 5D Convolution Implementation

Since PyTorch doesn't natively support 5D convolutions, we implement them using a double hierarchical approach:

### Double Hierarchical Convolution Strategy

The 5D autoencoder uses a hierarchical approach:

1. **Double Hierarchy**: 
   - 5D convolutions use 4D convolutions
   - 4D convolutions use 3D convolutions (PyTorch native)

2. **Dimension Processing Order**:
   - First process the channel dimension (C)
   - Then process the time dimension (T)
   - Finally process the spatial dimensions (D, H, W) using native 3D convolutions

3. **Sliding Window Approach**: For each higher dimension, we:
   - Take 3 consecutive slices (e.g., c, c+1, c+2)
   - Apply separate convolutions to each slice
   - Average the results to create one output slice
   - Repeat with stride to reduce the dimension

### Implementation Details

The `Conv5d` class processes the channel dimension by:

### Conv4d Class

Similarly, the `Conv4d` class implements 4D convolution by:
1. Processing the 4th dimension (time) using multiple 3D convolutions
2. For each slice in the 4th dimension, applying a 3D convolution
3. Combining the results with a weighted average
4. Stacking the outputs along the 4th dimension

### Transposed Convolutions

The transposed convolution classes (`ConvTranspose5d` and `ConvTranspose4d`) follow the same hierarchical pattern but use transposed convolutions to upsample the data.

## Dimensionality Reduction

Through the encoder, the 5D input is progressively reduced in spatial dimensions:
- Each convolution with stride 2 halves the size in each dimension
- After 3 layers, the spatial dimensions are reduced by a factor of 8
- The number of channels increases and then decreases, forming a bottleneck

## Training Process

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer with learning rate 0.001
- 500 training epochs

### Hierarchical Convolution Strategy

The 5D autoencoder extends our hierarchical approach even further:

1. **Double Hierarchy**: We implement 5D convolutions using 4D convolutions, which themselves use 3D convolutions
2. **Dimension Processing Order**: We first process the channel dimension (c_dim), then for each channel slice, we process the time dimension
3. **Weighted Combinations**: At each level, we combine results from multiple slices using weighted averages
4. **Nested Sliding Windows**: We slide windows along both the channel and time dimensions

This nested hierarchical approach allows us to build arbitrarily high-dimensional convolutions while still leveraging PyTorch's built-in operations at the core.