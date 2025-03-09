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


# Load and prepare the image
print("Loading Pokémon spritesheet...")
image_path = '../data/pokemon_151.png'
image = Image.open(image_path)  # Keep RGBA colors

# Get image dimensions
img_width = image.width
img_height = image.height
print(f"Spritesheet dimensions: {img_width}x{img_height}")

# Define dimensions for the spritesheet
num_cols = 15  # 15 sprites across
num_rows = 11  # 11 sprites high (with last row having only 1)
sprite_width = img_width // num_cols
sprite_height = img_height // num_rows
print(f"Individual sprite dimensions: {sprite_width}x{sprite_height}")

# Extract individual Pokémon sprites
print("Extracting individual Pokémon sprites...")
pokemon_sprites = []
for row in range(num_rows):
    for col in range(num_cols):
        # Skip empty spaces in the last row (only Mew is present)
        if row == 10 and col > 0:
            continue
            
        left = col * sprite_width
        upper = row * sprite_height
        right = left + sprite_width
        lower = upper + sprite_height
        
        sprite = image.crop((left, upper, right, lower))
        pokemon_sprites.append(sprite)

print(f"Extracted {len(pokemon_sprites)} Pokémon sprites")

# Convert sprites to tensors
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to power of 2 for easier conv/deconv
    transforms.ToTensor(),
])

# Use all Pokémon for training
input_tensors = torch.stack([transform(sprite) for sprite in pokemon_sprites])
print(f"Input tensor shape: {input_tensors.shape}")  # Should show the channel count

# Define 2D Autoencoder for RGBA images
class Autoencoder2D(nn.Module):
    def __init__(self, in_channels=4):
        super(Autoencoder2D, self).__init__()
        
        # Encoder - with dynamic input channels (4 for RGBA)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder - output same number of channels as input
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Sigmoid to keep values between 0 and 1
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Create the 2D autoencoder with the correct number of input channels
num_channels = input_tensors.shape[1]  # Get the actual number of channels
print(f"Creating 2D autoencoder model for Pokémon sprites with {num_channels} channels...")
autoencoder = Autoencoder2D(in_channels=num_channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training
print("Starting training...")
num_epochs = 1500  # Increased epochs for better quality and color preservation
losses = []
training_log = []  # For CSV logging

start_time = time.time()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(input_tensors)
    loss = criterion(outputs, input_tensors)
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
torch.save(autoencoder.state_dict(), 'autoencoder2d_pokemon.pt')

# Save training log to CSV
print("Saving training log to CSV...")
with open('training_log.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Loss'])  # Header
    csv_writer.writerows(training_log)

# Plot the loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('2D Pokémon Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()  # Close the figure to free memory

# Visualize original and reconstruction for a sample of Pokémon
with torch.no_grad():
    reconstructed = autoencoder(input_tensors)

# Select a subset of Pokémon to display (iconic ones from different generations)
# Adjust indices to stay within the range of available Pokémon
sample_indices = [0, 3, 6, 24, 54, 93, 129, 150]  # Bulbasaur, Charmander, Squirtle, Pikachu, etc.
sample_indices = [min(idx, len(pokemon_sprites) - 1) for idx in sample_indices]  # Ensure within bounds
num_samples = len(sample_indices)

# Plot original and reconstructed images for the sample
plt.figure(figsize=(16, 4))
for i, idx in enumerate(sample_indices):
    # Original
    plt.subplot(2, num_samples, i + 1)
    # Use proper normalization to preserve colors
    img = input_tensors[idx].permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.title(f'Original #{idx+1}')
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(2, num_samples, i + 1 + num_samples)
    # Use proper normalization to preserve colors
    recon_img = reconstructed[idx].permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(recon_img)
    plt.title(f'Reconstructed #{idx+1}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('reconstructed_samples.png')
plt.close()  # Close the figure to free memory

# Create a grid of all reconstructed Pokémon (15x11 grid like the original spritesheet)
plt.figure(figsize=(20, 16))
pokemon_count = 0
for row in range(num_rows):
    for col in range(num_cols):
        # Skip empty spaces in the last row (only Mew is present)
        if row == 10 and col > 0:
            continue
            
        plt.subplot(num_rows, num_cols, row * num_cols + col + 1)
        # Use proper normalization to preserve colors
        recon_img = reconstructed[pokemon_count].permute(1, 2, 0).cpu().detach().numpy()
        plt.imshow(recon_img)
        plt.axis('off')
        pokemon_count += 1

plt.tight_layout()
plt.savefig('reconstructed_all.png')
plt.close()  # Close the figure to free memory

# Visualize the encoded representation for a few selected Pokémon
with torch.no_grad():
    encoded = autoencoder.get_encoded(input_tensors[sample_indices])

# Plot the encoded representation (feature maps) for the sample Pokémon
fig, axs = plt.subplots(len(sample_indices), 8, figsize=(16, 2*len(sample_indices)))
for i in range(len(sample_indices)):
    for j in range(8):  # Show first 8 feature maps (out of 16)
        axs[i, j].imshow(encoded[i, j].cpu().detach().numpy(), cmap='viridis')
        if i == 0:
            axs[i, j].set_title(f'Feature {j+1}')
        if j == 0:
            axs[i, j].set_ylabel(f'Pokémon #{sample_indices[i]+1}')
        axs[i, j].axis('off')
plt.tight_layout()
plt.savefig('feature_maps.png')
plt.close()  # Close the figure to free memory

print("All done! Check the output files in the current directory.")

