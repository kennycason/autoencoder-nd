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

# Create a 4D representation by adding two extra dimensions
print("Creating 4D representation...")
depth = 8  # Number of depth slices
time_steps = 8  # Increased time steps to ensure enough for convolution
input_4d = torch.zeros(1, 1, time_steps, depth, height, width)  # Shape: [1, 1, T, D, H, W]

# Fill the 4D tensor with the 2D image repeated across dimensions
for t in range(time_steps):
    for d in range(depth):
        input_4d[0, 0, t, d] = input_tensor[0, 0]

print(f"4D input shape: {input_4d.shape}")

# Simplified 4D convolution implementation
class Conv4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv4d, self).__init__()
        # Use multiple 3D convolutions to simulate 4D
        self.conv_t1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_t2 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_t3 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.stride = stride
        
    def forward(self, x):
        # x shape: [batch, channels, time, depth, height, width]
        batch_size, channels, time_steps, depth, height, width = x.shape
        
        # Process different time slices
        t_out = []
        for t in range(0, time_steps, self.stride):
            if t+2 < time_steps:  # Ensure we have enough time steps
                # Weighted combination of 3 consecutive time slices
                x1 = self.conv_t1(x[:, :, t, :, :, :])
                x2 = self.conv_t2(x[:, :, t+1, :, :, :])
                x3 = self.conv_t3(x[:, :, t+2, :, :, :])
                t_out.append((x1 + x2 + x3) / 3.0)
        
        if not t_out:  # Handle case where time_steps is too small
            # Just process the first time slice
            t_out.append(self.conv_t1(x[:, :, 0, :, :, :]))
        
        # Stack along time dimension
        out = torch.stack(t_out, dim=2)
        return out

# Simplified 4D transposed convolution
class ConvTranspose4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
        super(ConvTranspose4d, self).__init__()
        # Use multiple 3D transposed convolutions
        self.conv_t1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_t2 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_t3 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.stride = stride
        self.out_channels = out_channels
        
    def forward(self, x):
        # x shape: [batch, channels, time, depth, height, width]
        batch_size, channels, time_steps, depth, height, width = x.shape
        
        # Process different time slices
        t_out = []
        for t in range(0, time_steps, self.stride):
            if t+2 < time_steps:  # Ensure we have enough time steps
                # Weighted combination of 3 consecutive time slices
                x1 = self.conv_t1(x[:, :, t, :, :, :])
                x2 = self.conv_t2(x[:, :, t+1, :, :, :])
                x3 = self.conv_t3(x[:, :, t+2, :, :, :])
                t_out.append((x1 + x2 + x3) / 3.0)
        
        if not t_out:  # Handle case where time_steps is too small
            # Just process the first time slice
            t_out.append(self.conv_t1(x[:, :, 0, :, :, :]))
        
        # Stack along time dimension
        out = torch.stack(t_out, dim=2)
        return out

# Define 4D Autoencoder with 3 layers in encoder and decoder
class Autoencoder4D(nn.Module):
    def __init__(self, time_steps, depth, height, width):
        super(Autoencoder4D, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            Conv4d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Conv4d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Conv4d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            ConvTranspose4d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            ConvTranspose4d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            ConvTranspose4d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.encoder(x)
        print(f"Encoded shape: {x.shape}")
        x = self.decoder(x)
        print(f"Output shape: {x.shape}")
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Create the 4D autoencoder
print("Creating 4D autoencoder model with 3 layers...")
autoencoder = Autoencoder4D(time_steps, depth, height, width)
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
    outputs = autoencoder(input_4d)
    
    # Ensure outputs and input_4d have the same shape for loss calculation
    # If they don't, we'll need to handle this differently
    if outputs.shape != input_4d.shape:
        print(f"Warning: Output shape {outputs.shape} doesn't match input shape {input_4d.shape}")
        
        # Extract a slice from the input that matches the output time dimension
        if outputs.size(2) < input_4d.size(2):
            # Take the first slice of the time dimension from input
            input_slice = input_4d[:, :, :outputs.size(2), :, :, :]
            loss = criterion(outputs, input_slice)
        else:
            # Repeat the output along time dimension to match input
            repeated_output = outputs.repeat(1, 1, input_4d.size(2) // outputs.size(2), 1, 1, 1)
            # If there's a remainder, pad with the last time slice
            if input_4d.size(2) % outputs.size(2) != 0:
                pad_size = input_4d.size(2) - repeated_output.size(2)
                pad = outputs[:, :, -1:].repeat(1, 1, pad_size, 1, 1, 1)
                repeated_output = torch.cat([repeated_output, pad], dim=2)
            loss = criterion(repeated_output, input_4d)
    else:
        loss = criterion(outputs, input_4d)
    
    loss.backward()
    optimizer.step()
    
    loss_value = loss.item()
    losses.append(loss_value)
    
    # Log the epoch and loss for CSV
    training_log.append([epoch, loss_value])
    
    if epoch % 50 == 0:
        elapsed = time.time() - start_time
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}, Time: {elapsed:.2f}s')

# Save the model
print("Saving model...")
torch.save(autoencoder.state_dict(), 'autoencoder4d.pt')

# Save training log to CSV
print("Saving training log to CSV...")
with open('training_log.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Loss'])  # Header
    csv_writer.writerows(training_log)

# Plot the loss curve
print("Plotting loss curve...")
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('4D Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()  # Close the figure to free memory

# Visualize original and reconstruction
print("Generating reconstruction...")
with torch.no_grad():
    reconstructed = autoencoder(input_4d)
    
    # Handle shape mismatch for visualization
    if reconstructed.shape != input_4d.shape:
        print(f"Adjusting reconstruction shape for visualization: {reconstructed.shape} -> {input_4d.shape}")
        
        # For visualization, we'll just repeat the output to match input dimensions
        if reconstructed.size(2) < input_4d.size(2):
            # Repeat along time dimension
            repeat_factor = input_4d.size(2) // reconstructed.size(2)
            reconstructed = reconstructed.repeat(1, 1, repeat_factor, 1, 1, 1)
            
            # If there's a remainder, pad with the last time slice
            if input_4d.size(2) % reconstructed.size(2) != 0:
                pad_size = input_4d.size(2) - reconstructed.size(2)
                pad = reconstructed[:, :, -1:].repeat(1, 1, pad_size, 1, 1, 1)
                reconstructed = torch.cat([reconstructed, pad], dim=2)

# Visualize middle slices of the 4D volumes
middle_time = time_steps // 2
middle_slice = depth // 2

# Original middle slice
orig_slice = input_4d[0, 0, middle_time, middle_slice].cpu().numpy()

# Reconstructed middle slice - ensure we're within bounds
recon_time = min(middle_time, reconstructed.size(2) - 1)
recon_slice = reconstructed[0, 0, recon_time, middle_slice].cpu().detach().numpy()

print("Plotting reconstruction comparison...")
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(orig_slice, cmap='gray')
axs[0].set_title('Original (Middle Slice)')
axs[1].imshow(recon_slice, cmap='gray')
axs[1].set_title('Reconstructed (Middle Slice)')
plt.savefig('reconstructed.png')
plt.close()  # Close the figure to free memory

# Visualize the encoded representation
print("Generating feature maps...")
with torch.no_grad():
    encoded = autoencoder.get_encoded(input_4d)

# Plot the encoded representation (feature maps)
if encoded.dim() >= 5:  # Check if we have enough dimensions
    middle_time_encoded = min(encoded.size(2) // 2, encoded.size(2) - 1)
    middle_depth_encoded = min(encoded.size(3) // 2, encoded.size(3) - 1)
    encoded_middle_slices = encoded[0, :, middle_time_encoded, middle_depth_encoded].detach().cpu().numpy()
    
    fig, axs = plt.subplots(1, min(4, encoded.size(1)), figsize=(12, 3))
    for i in range(min(4, encoded.size(1))):
        if isinstance(axs, np.ndarray):
            axs[i].imshow(encoded_middle_slices[i], cmap='viridis')
            axs[i].set_title(f'Feature Map {i+1} (Middle Slice)')
            axs[i].axis('off')
        else:
            axs.imshow(encoded_middle_slices[i], cmap='viridis')
            axs.set_title(f'Feature Map {i+1} (Middle Slice)')
            axs.axis('off')
    plt.tight_layout()
    plt.savefig('feature_map.png')
    plt.close()  # Close the figure to free memory

# Create a 3D visualization of one feature map (from the 4D space)
print("Starting 3D visualization of a 4D slice (this may take a while)...")
from mpl_toolkits.mplot3d import Axes3D

if encoded.dim() >= 5:  # Check if we have enough dimensions
    # Select one feature map and one time slice
    feature_idx = 0
    time_idx = min(encoded.size(2) // 2, encoded.size(2) - 1)
    feature_volume = encoded[0, feature_idx, time_idx].detach().cpu().numpy()

    # Create a 3D plot - use a smaller subset of the data to avoid hanging
    print("Downsampling volume for visualization...")
    # Downsample the volume to make visualization faster
    downsample_factor = 4  # Increased downsampling for faster rendering
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
    ax.set_title(f'3D Visualization of 4D Feature Map {feature_idx+1} (Time Slice {time_idx}, Downsampled)')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Height')
    ax.set_zlabel('Width')

    print("Saving 3D visualization...")
    plt.savefig('feature_volume.png')
    plt.close()  # Close the figure to free memory

print("All done! Check the output files.")
