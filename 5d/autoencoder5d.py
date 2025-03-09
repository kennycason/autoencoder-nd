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

# Create a 5D representation by adding three extra dimensions
print("Creating 5D representation...")
depth = 8  # Number of depth slices
time_steps = 8  # Time steps
channels = 4  # Additional dimension (could represent different "channels" or "features")
input_5d = torch.zeros(1, 1, channels, time_steps, depth, height, width)  # Shape: [1, 1, C, T, D, H, W]

# Fill the 5D tensor with the 2D image repeated across dimensions
for c in range(channels):
    for t in range(time_steps):
        for d in range(depth):
            input_5d[0, 0, c, t, d] = input_tensor[0, 0]

print(f"5D input shape: {input_5d.shape}")

# Simplified 5D convolution implementation
class Conv5d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv5d, self).__init__()
        # Use multiple 4D convolutions to simulate 5D
        self.conv_c1 = Conv4d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_c2 = Conv4d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_c3 = Conv4d(in_channels, out_channels, kernel_size, stride, padding)
        self.stride = stride
        
    def forward(self, x):
        # x shape: [batch, channels, c_dim, time, depth, height, width]
        batch_size, channels, c_dim, time_steps, depth, height, width = x.shape
        
        # Process different channel slices
        c_out = []
        for c in range(0, c_dim, self.stride):
            if c+2 < c_dim:  # Ensure we have enough channel steps
                # Weighted combination of 3 consecutive channel slices
                x1 = self.conv_c1(x[:, :, c, :, :, :, :])
                x2 = self.conv_c2(x[:, :, c+1, :, :, :, :])
                x3 = self.conv_c3(x[:, :, c+2, :, :, :, :])
                c_out.append((x1 + x2 + x3) / 3.0)
        
        if not c_out:  # Handle case where c_dim is too small
            # Just process the first channel slice
            c_out.append(self.conv_c1(x[:, :, 0, :, :, :, :]))
        
        # Stack along channel dimension
        out = torch.stack(c_out, dim=2)
        return out

# Simplified 4D convolution implementation (used by Conv5d)
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

# Simplified 5D transposed convolution
class ConvTranspose5d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
        super(ConvTranspose5d, self).__init__()
        # Use multiple 4D transposed convolutions
        self.conv_c1 = ConvTranspose4d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_c2 = ConvTranspose4d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_c3 = ConvTranspose4d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.stride = stride
        self.out_channels = out_channels
        
    def forward(self, x):
        # x shape: [batch, channels, c_dim, time, depth, height, width]
        batch_size, channels, c_dim, time_steps, depth, height, width = x.shape
        
        # Process different channel slices
        c_out = []
        for c in range(0, c_dim, self.stride):
            if c+2 < c_dim:  # Ensure we have enough channel steps
                # Weighted combination of 3 consecutive channel slices
                x1 = self.conv_c1(x[:, :, c, :, :, :, :])
                x2 = self.conv_c2(x[:, :, c+1, :, :, :, :])
                x3 = self.conv_c3(x[:, :, c+2, :, :, :, :])
                c_out.append((x1 + x2 + x3) / 3.0)
        
        if not c_out:  # Handle case where c_dim is too small
            # Just process the first channel slice
            c_out.append(self.conv_c1(x[:, :, 0, :, :, :, :]))
        
        # Stack along channel dimension
        out = torch.stack(c_out, dim=2)
        return out

# Simplified 4D transposed convolution (used by ConvTranspose5d)
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

# Define 5D Autoencoder with 3 layers in encoder and decoder
class Autoencoder5D(nn.Module):
    def __init__(self, channels, time_steps, depth, height, width):
        super(Autoencoder5D, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            Conv5d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Conv5d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Conv5d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            ConvTranspose5d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            ConvTranspose5d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            ConvTranspose5d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
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

# Create the 5D autoencoder
print("Creating 5D autoencoder model with 3 layers...")
autoencoder = Autoencoder5D(channels, time_steps, depth, height, width)
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
    outputs = autoencoder(input_5d)
    
    # Ensure outputs and input_5d have the same shape for loss calculation
    if outputs.shape != input_5d.shape:
        print(f"Warning: Output shape {outputs.shape} doesn't match input shape {input_5d.shape}")
        
        # Extract a slice from the input that matches the output dimensions
        if outputs.size(2) < input_5d.size(2) or outputs.size(3) < input_5d.size(3):
            # Take slices that match the output dimensions
            input_slice = input_5d[:, :, :outputs.size(2), :outputs.size(3), :, :, :]
            loss = criterion(outputs, input_slice)
        else:
            # Repeat the output along dimensions to match input
            # This is a simplified approach - in practice, you might need more complex handling
            loss = criterion(outputs[:, :, :input_5d.size(2), :input_5d.size(3), :, :, :], input_5d)
    else:
        loss = criterion(outputs, input_5d)
    
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
torch.save(autoencoder.state_dict(), 'autoencoder5d.pt')

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
plt.title('5D Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()  # Close the figure to free memory

# Visualize original and reconstruction
print("Generating reconstruction...")
with torch.no_grad():
    reconstructed = autoencoder(input_5d)
    
    # Handle shape mismatch for visualization
    if reconstructed.shape != input_5d.shape:
        print(f"Adjusting reconstruction shape for visualization: {reconstructed.shape} -> {input_5d.shape}")
        
        # For visualization, we'll just use the available dimensions
        # This is a simplified approach - in practice, you might need more complex handling

# Visualize middle slices of the 5D volumes
middle_channel = min(channels // 2, reconstructed.size(2) - 1)
middle_time = min(time_steps // 2, reconstructed.size(3) - 1)
middle_slice = depth // 2

# Original middle slice
orig_slice = input_5d[0, 0, middle_channel, middle_time, middle_slice].cpu().numpy()

# Reconstructed middle slice - ensure we're within bounds
recon_channel = min(middle_channel, reconstructed.size(2) - 1)
recon_time = min(middle_time, reconstructed.size(3) - 1)
recon_slice = reconstructed[0, 0, recon_channel, recon_time, middle_slice].cpu().detach().numpy()

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
    encoded = autoencoder.get_encoded(input_5d)

# Plot the encoded representation (feature maps)
if encoded.dim() >= 7:  # Check if we have enough dimensions
    middle_channel_encoded = min(encoded.size(2) // 2, encoded.size(2) - 1)
    middle_time_encoded = min(encoded.size(3) // 2, encoded.size(3) - 1)
    middle_depth_encoded = min(encoded.size(4) // 2, encoded.size(4) - 1)
    
    encoded_middle_slices = encoded[0, :, middle_channel_encoded, middle_time_encoded, middle_depth_encoded].detach().cpu().numpy()
    
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

# Create a 3D visualization of one feature map (from the 5D space)
print("Starting 3D visualization of a 5D slice (this may take a while)...")
from mpl_toolkits.mplot3d import Axes3D

if encoded.dim() >= 7:  # Check if we have enough dimensions
    # Select one feature map and one time slice
    feature_idx = 0
    channel_idx = min(encoded.size(2) // 2, encoded.size(2) - 1)
    time_idx = min(encoded.size(3) // 2, encoded.size(3) - 1)
    
    feature_volume = encoded[0, feature_idx, channel_idx, time_idx].detach().cpu().numpy()

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
    ax.set_title(f'3D Visualization of 5D Feature Map\n(Channel {channel_idx}, Time {time_idx}, Downsampled)')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Height')
    ax.set_zlabel('Width')

    print("Saving 3D visualization...")
    plt.savefig('feature_volume_5d.png')
    plt.close()  # Close the figure to free memory

print("All done! Check the output files.")
