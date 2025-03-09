# Autoencoder ND - 1-6D Convolutional Layers

This project explores how convolutional autoencoders can be implemented with layers of different dimensionalities, from 1D to 6D. While we always start with the same 2D image data, we process it through neural networks whose internal layers operate on tensors of increasing dimensionality.

This is an exploration with Cursor + Claude 3.7

## Project Overview

A autoencoder is a type of neural network that consists of two parts:
1. An **encoder** (a convolutional neural network that compresses the input)
2. A **decoder** (a transposed convolutional neural network that reconstructs the input)

Each higher-dimensional layer creates more connections between neurons, allowing for more complex pattern recognition across multiple dimensions simultaneously.

We start with a standard 2D grayscale image and process it using autoencoders with different layer dimensionalities:

1. **1D Autoencoder**: Flattens the 2D image into a 1D sequence for processing with 1D layers
2. **2D Autoencoder**: Processes the image directly with 2D layers
3. **3D Autoencoder**: Expands the image into a 3D volume for processing with 3D layers
4. **4D Autoencoder**: Creates a 4D representation for processing with 4D layers
5. **5D Autoencoder**: Creates a 5D representation for processing with 5D layers
6. **6D Autoencoder**: Creates a 6D representation for processing with 6D layers

Each implementation maintains the same architectural pattern:
- **Encoder**: 3 convolutional layers with stride 2 (1→8→4→4 channels)
- **Decoder**: 3 transposed convolutional layers with stride 2 (4→4→8→1 channels)

## Hierarchical Convolution Approach

For dimensions higher than 3D, we implement a hierarchical convolution approach:

- **3D and below**: Use PyTorch's native convolution and transposed convolution operations
- **4D**: Implement 4D convolutions using multiple 3D convolutions
- **5D**: Implement 5D convolutions using multiple 4D convolutions
- **6D**: Implement 6D convolutions using multiple 5D convolutions

This hierarchical approach demonstrates how convolution operations can be extended to arbitrarily high dimensions.

## Results Visualization

### Training Loss Comparison

Each model was trained for 500 epochs on the same input image. Below is the combined training loss curve for all dimensions:

![Combined Training Loss](training_loss.png)

Below are the individual training loss curves:

<div style="display: flex; flex-wrap: wrap;">
  <div style="flex: 50%;">
    <h4>1D Autoencoder</h4>
    <img src="1d/training_loss.png" alt="1D Training Loss" style="width: 100%;">
  </div>
  <div style="flex: 50%;">
    <h4>2D Autoencoder</h4>
    <img src="2d/training_loss.png" alt="2D Training Loss" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap;">
  <div style="flex: 50%;">
    <h4>3D Autoencoder</h4>
    <img src="3d/training_loss.png" alt="3D Training Loss" style="width: 100%;">
  </div>
  <div style="flex: 50%;">
    <h4>4D Autoencoder</h4>
    <img src="4d/training_loss.png" alt="4D Training Loss" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap;">
  <div style="flex: 50%;">
    <h4>5D Autoencoder</h4>
    <img src="5d/training_loss.png" alt="5D Training Loss" style="width: 100%;">
  </div>
  <div style="flex: 50%;">
    <h4>6D Autoencoder</h4>
    <img src="6d/training_loss.png" alt="6D Training Loss" style="width: 100%;">
  </div>
</div>

### Reconstruction Quality

Each model attempts to reconstruct the original image. Below are the reconstruction results:

<div style="display: flex; flex-wrap: wrap;">
  <div style="flex: 50%;">
    <h4>1D Autoencoder</h4>
    <img src="1d/reconstructed.png" alt="1D Reconstruction" style="width: 100%;">
  </div>
  <div style="flex: 50%;">
    <h4>2D Autoencoder</h4>
    <img src="2d/reconstructed.png" alt="2D Reconstruction" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap;">
  <div style="flex: 50%;">
    <h4>3D Autoencoder</h4>
    <img src="3d/reconstructed.png" alt="3D Reconstruction" style="width: 100%;">
  </div>
  <div style="flex: 50%;">
    <h4>4D Autoencoder</h4>
    <img src="4d/reconstructed.png" alt="4D Reconstruction" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap;">
  <div style="flex: 50%;">
    <h4>5D Autoencoder</h4>
    <img src="5d/reconstructed.png" alt="5D Reconstruction" style="width: 100%;">
  </div>
  <div style="flex: 50%;">
    <h4>6D Autoencoder</h4>
    <img src="6d/reconstructed.png" alt="6D Reconstruction" style="width: 100%;">
  </div>
</div>
