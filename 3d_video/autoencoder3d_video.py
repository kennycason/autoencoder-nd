import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import time
import csv
import os
import glob
from torchvision.utils import make_grid
import torch.nn.functional as F

# Configuration
frame_dir = '../data/video/frames'
target_size = (270, 480)  # Scaled down from 1080x1920 by factor of 4
num_frames_to_use = 16  # Number of frames to use in each 3D volume
batch_size = 1
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# Create output directories
os.makedirs('frame_comparisons', exist_ok=True)

# Load and prepare the frames
print("Loading video frames...")
frame_paths = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.jpg')))
print(f"Found {len(frame_paths)} frames")

# Take a subset of frames at regular intervals to create our 3D volume
total_frames = len(frame_paths)
frame_indices = np.linspace(0, total_frames-1, num_frames_to_use, dtype=int)
selected_frames = [frame_paths[i] for i in frame_indices]

print(f"Selected {len(selected_frames)} frames at indices: {frame_indices}")

# Transform to resize and normalize (keeping color)
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
])

# Load the selected frames
frames = []
for frame_path in selected_frames:
    img = Image.open(frame_path)
    img_tensor = transform(img)
    frames.append(img_tensor)

# Stack frames to create a 3D volume [1, C, D, H, W]
# Where C is the number of channels (3 for RGB)
input_3d = torch.stack(frames, dim=1)  # Stack along dimension 1 (channels become frames)
input_3d = input_3d.unsqueeze(0)  # Add batch dimension
print(f"3D input tensor shape: {input_3d.shape}")

# Define 3D Autoencoder with improved architecture
class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=3):
        super(Autoencoder3D, self).__init__()
        
        # Encoder - with more feature maps and an additional layer
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder - with corresponding structure
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Tanh instead of Sigmoid for better color reproduction
        )
        
        # Store input shape for resizing output
        self.input_shape = None
    
    def forward(self, x):
        # Store input shape
        if self.input_shape is None:
            self.input_shape = x.shape
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        # Resize output to match input dimensions exactly
        if x.shape != self.input_shape:
            x = F.interpolate(x, size=(self.input_shape[2], self.input_shape[3], self.input_shape[4]), 
                             mode='trilinear', align_corners=False)
        
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Create the 3D autoencoder
print("Creating 3D autoencoder model...")
autoencoder = Autoencoder3D(in_channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Move input to device
input_3d = input_3d.to(device)

# Training
print("Starting training...")
losses = []
training_log = []

start_time = time.time()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(input_3d)
    loss = criterion(outputs, input_3d)
    loss.backward()
    optimizer.step()
    
    loss_value = loss.item()
    losses.append(loss_value)
    training_log.append([epoch, loss_value])
    
    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss_value:.6f}, Time: {elapsed:.2f}s")

# Save the training loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()

# Save the training log to CSV
with open('training_log.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss'])
    writer.writerows(training_log)

print("Training complete!")

# Visualize original and reconstruction
with torch.no_grad():
    reconstructed = autoencoder(input_3d)

# Visualize each frame and its reconstruction
for i in range(num_frames_to_use):
    orig_frame = input_3d[0, :, i].permute(1, 2, 0).cpu().numpy()  # Change to HWC format for plotting
    recon_frame = reconstructed[0, :, i].permute(1, 2, 0).cpu().detach().numpy()
    
    # Clip values to valid range for imshow
    orig_frame = np.clip(orig_frame, 0, 1)
    recon_frame = np.clip(recon_frame, 0, 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(orig_frame)
    axs[0].set_title(f'Original Frame {i+1}')
    axs[0].axis('off')
    
    axs[1].imshow(recon_frame)
    axs[1].set_title(f'Reconstructed Frame {i+1}')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'frame_comparisons/frame_{i+1}_comparison.png')
    plt.close()

# Create a grid visualization of all frames (showing only a subset if there are many)
max_display = min(8, num_frames_to_use)  # Display at most 8 frames in the grid
display_indices = np.linspace(0, num_frames_to_use-1, max_display, dtype=int)

fig, axs = plt.subplots(2, max_display, figsize=(20, 6))

for i, idx in enumerate(display_indices):
    # Original frames on top row
    orig_frame = input_3d[0, :, idx].permute(1, 2, 0).cpu().numpy()
    axs[0, i].imshow(np.clip(orig_frame, 0, 1))
    axs[0, i].set_title(f'Original {idx+1}')
    axs[0, i].axis('off')
    
    # Reconstructed frames on bottom row
    recon_frame = reconstructed[0, :, idx].permute(1, 2, 0).cpu().detach().numpy()
    axs[1, i].imshow(np.clip(recon_frame, 0, 1))
    axs[1, i].set_title(f'Recon {idx+1}')
    axs[1, i].axis('off')

plt.tight_layout()
plt.savefig('all_frames_comparison.png')
plt.close()

# Visualize the encoded representation
with torch.no_grad():
    encoded = autoencoder.get_encoded(input_3d)

print(f"Encoded shape: {encoded.shape}")

# Visualize feature maps for a middle frame
# Get the actual depth dimension size
depth_size = encoded.shape[2]
middle_frame_idx = depth_size // 2  # This will be 1 if depth_size is 2
num_features = min(4, encoded.shape[1])  # Display at most 4 feature maps

# Create a grid of feature maps for the middle frame
fig, axs = plt.subplots(1, num_features, figsize=(15, 3))
for i in range(num_features):
    feature_map = encoded[0, i, middle_frame_idx].cpu().detach().numpy()
    if num_features == 1:
        axs.imshow(feature_map, cmap='viridis')
        axs.set_title(f'Feature {i+1}')
        axs.axis('off')
    else:
        axs[i].imshow(feature_map, cmap='viridis')
        axs[i].set_title(f'Feature {i+1}')
        axs[i].axis('off')

plt.tight_layout()
plt.savefig('feature_maps_middle_frame.png')
plt.close()

# Create a 3D visualization of one feature map
print("Creating 3D visualization...")
from mpl_toolkits.mplot3d import Axes3D

# Select one feature map
feature_idx = 0
feature_volume = encoded[0, feature_idx].detach().cpu().numpy()

# Downsample for visualization
downsample_factor = 2
downsampled_volume = feature_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
print(f"Original volume shape: {feature_volume.shape}, Downsampled: {downsampled_volume.shape}")

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create coordinate matrices
x, y, z = np.indices(downsampled_volume.shape)

# Use a threshold to make the plot clearer
threshold = downsampled_volume.mean() + 0.5 * downsampled_volume.std()
voxels = downsampled_volume > threshold

# Plot voxels
ax.voxels(voxels, facecolors='red', alpha=0.3, edgecolor=None)
ax.set_title(f'3D Visualization of Feature Map {feature_idx+1}')
ax.set_xlabel('Time (Frames)')
ax.set_ylabel('Height')
ax.set_zlabel('Width')

plt.savefig('feature_volume_3d.png')
plt.close()

# Create a temporal visualization of a feature map
feature_idx = 0
plt.figure(figsize=(15, 5))
for i in range(min(8, encoded.shape[2])):  # For each depth slice (time), up to 8
    plt.subplot(1, min(8, encoded.shape[2]), i+1)
    plt.imshow(encoded[0, feature_idx, i].cpu().detach().numpy(), cmap='viridis')
    plt.title(f'Time {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('feature_map_over_time.png')
plt.close()

print("All done! Check the output files.")

