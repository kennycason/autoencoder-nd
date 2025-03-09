import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import time
import csv

# Load and prepare the image
print("Loading image...")
image_path = '../data/meshman.png'
image = Image.open(image_path).convert('L')  # convert to grayscale
transform = transforms.Compose([
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)  # shape [1, 1, H, W]

# Get image dimensions
_, _, height, width = input_tensor.shape
print(f"Image dimensions: {height}x{width}")

# Create a 3D representation by adding a depth dimension
print("Creating 3D representation...")
depth = 8  # Number of depth slices
input_3d = torch.zeros(1, 1, depth, height, width)  # Shape: [1, 1, D, H, W]
for d in range(depth):
    input_3d[0, 0, d] = input_tensor[0, 0]

# Define 3D Autoencoder with 3 layers in encoder and decoder
class Autoencoder3D(nn.Module):
    def __init__(self, depth, height, width):
        super(Autoencoder3D, self).__init__()
        
        # Enhanced encoder - three 3D convolutional layers (matching 2D architecture)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate encoded dimensions
        self.encoded_depth = depth // 8  # After 3 layers with stride 2
        self.encoded_height = height // 8
        self.encoded_width = width // 8
        
        # Enhanced decoder - three 3D transposed convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Create the 3D autoencoder
print("Creating enhanced 3D autoencoder model with 3 layers...")
autoencoder = Autoencoder3D(depth, height, width)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training
print("Starting training...")
num_epochs = 500
losses = []
training_log = []  # For CSV logging

start_time = time.time()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(input_3d)
    loss = criterion(outputs, input_3d)
    loss.backward()
    optimizer.step()
    
    loss_value = loss.item()
    losses.append(loss_value)
    
    # Log the epoch and loss for CSV
    training_log.append([epoch, loss_value])
    
    if epoch % 50 == 0:
        elapsed = time.time() - start_time
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss_value:.6f}, Time: {elapsed:.2f}s')

# Save the model
print("Saving model...")
torch.save(autoencoder.state_dict(), 'autoencoder3d.pt')

# Save training log to CSV
print("Saving training log to CSV...")
with open('training_log.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Loss'])  # Header
    csv_writer.writerows(training_log)

# Plot the loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('3D Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()  # Close the figure to free memory

# Visualize original and reconstruction
with torch.no_grad():
    reconstructed = autoencoder(input_3d)

# Visualize middle slices of the 3D volumes
middle_slice = depth // 2

# Original middle slice
orig_slice = input_3d[0, 0, middle_slice].cpu().numpy()

# Reconstructed middle slice
recon_slice = reconstructed[0, 0, middle_slice].cpu().detach().numpy()

# Ensure the slices are 2D arrays
if orig_slice.ndim == 1:
    orig_slice = orig_slice.reshape(int(np.sqrt(orig_slice.shape[0])), -1)
if recon_slice.ndim == 1:
    recon_slice = recon_slice.reshape(int(np.sqrt(recon_slice.shape[0])), -1)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(orig_slice, cmap='gray')
axs[0].set_title('Original (Middle Slice)')
axs[1].imshow(recon_slice, cmap='gray')
axs[1].set_title('Reconstructed (Middle Slice)')
plt.savefig('reconstructed.png')
plt.close()  # Close the figure to free memory

# Visualize the encoded representation (middle slice of each feature map)
with torch.no_grad():
    encoded = autoencoder.get_encoded(input_3d)

# Plot the encoded representation (feature maps)
encoded_middle_slices = encoded[0, :, encoded.size(2)//2].detach().cpu().numpy()
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axs[i].imshow(encoded_middle_slices[i], cmap='viridis')
    axs[i].set_title(f'Feature Map {i+1} (Middle Slice)')
    axs[i].axis('off')
plt.tight_layout()
plt.savefig('feature_map.png')
plt.close()  # Close the figure to free memory

# Create a 3D visualization of one feature map
print("Starting 3D visualization (this may take a while)...")
from mpl_toolkits.mplot3d import Axes3D

# Select one feature map
feature_idx = 0
feature_volume = encoded[0, feature_idx].detach().cpu().numpy()

# Create a 3D plot - use a smaller subset of the data to avoid hanging
print("Downsampling volume for visualization...")
# Downsample the volume to make visualization faster
downsample_factor = 8  # Increased downsampling for faster rendering
downsampled_volume = feature_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
print(f"Original volume shape: {feature_volume.shape}, Downsampled: {downsampled_volume.shape}")

# Create a 3D plot
print("Creating 3D plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create coordinate matrices for the downsampled volume
print("Creating coordinate matrices...")
x, y, z = np.indices(downsampled_volume.shape)

# Use a threshold to make the plot clearer
print("Applying threshold...")
threshold = downsampled_volume.mean() + 0.5 * downsampled_volume.std()
voxels = downsampled_volume > threshold

# Plot voxels with reduced complexity
print("Rendering voxels (this is the slow part)...")
ax.voxels(voxels, facecolors='red', alpha=0.3, edgecolor=None)
ax.set_title(f'3D Visualization of Feature Map {feature_idx+1} (Downsampled)')
ax.set_xlabel('Depth')
ax.set_ylabel('Height')
ax.set_zlabel('Width')

print("Saving 3D visualization...")
plt.savefig('feature_volume.png')
plt.close()  # Close the figure to free memory

print("All done! Check the output files.")
