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

# Flatten the 2D image into a 1D array
print("Flattening 2D image to 1D...")
input_1d = input_tensor.view(1, 1, -1)  # Shape: [1, 1, H*W]
print(f"1D input shape: {input_1d.shape}")

# Define 1D Autoencoder
class Autoencoder1D(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder1D, self).__init__()
        
        # Calculate sizes for each layer to maintain consistency with other dimensions
        self.layer1_size = input_size // 2
        self.layer2_size = self.layer1_size // 2
        self.layer3_size = self.layer2_size // 2
        
        # Encoder - three 1D convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder - three 1D transposed convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Create the 1D autoencoder
print("Creating 1D autoencoder model...")
autoencoder = Autoencoder1D(input_size=height*width)
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
    outputs = autoencoder(input_1d)
    loss = criterion(outputs, input_1d)
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
torch.save(autoencoder.state_dict(), 'autoencoder1d.pt')

# Save training log to CSV
print("Saving training log to CSV...")
with open('training_log.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Loss'])  # Header
    csv_writer.writerows(training_log)

# Plot the loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('1D Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()  # Close the figure to free memory

# Visualize original and reconstruction
with torch.no_grad():
    reconstructed_1d = autoencoder(input_1d)
    # Reshape back to 2D for visualization
    reconstructed = reconstructed_1d.view(1, 1, height, width)

# Plot original and reconstructed images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(input_tensor[0, 0].cpu().numpy(), cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed[0, 0].cpu().detach().numpy(), cmap='gray')
plt.title('Reconstructed')
plt.axis('off')

plt.savefig('reconstructed.png')
plt.close()  # Close the figure to free memory

# Visualize the encoded representation
with torch.no_grad():
    encoded = autoencoder.get_encoded(input_1d)

# Plot the encoded representation (feature maps)
# For 1D, we'll visualize as line plots
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
axs = axs.flatten()
for i in range(4):
    # Take a subset of points to make the plot clearer
    sample_points = encoded[0, i].cpu().detach().numpy()
    if len(sample_points) > 1000:
        sample_points = sample_points[::len(sample_points)//1000]
    axs[i].plot(sample_points)
    axs[i].set_title(f'Feature Map {i+1}')
    axs[i].grid(True)
plt.tight_layout()
plt.savefig('feature_map.png')
plt.close()  # Close the figure to free memory

# Also visualize the encoded representation as a heatmap
plt.figure(figsize=(12, 4))
encoded_np = encoded[0].cpu().detach().numpy()
# Normalize for better visualization
encoded_np = (encoded_np - encoded_np.min()) / (encoded_np.max() - encoded_np.min() + 1e-8)
plt.imshow(encoded_np, aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Activation')
plt.title('1D Encoded Representation (4 feature maps)')
plt.xlabel('Position in Sequence')
plt.ylabel('Feature Map')
plt.savefig('feature_heatmap.png')
plt.close()  # Close the figure to free memory

print("All done! Check the output files.")
