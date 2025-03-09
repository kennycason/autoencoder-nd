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

# Define 2D Autoencoder
class Autoencoder2D(nn.Module):
    def __init__(self):
        super(Autoencoder2D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Create the 2D autoencoder
print("Creating 2D autoencoder model...")
autoencoder = Autoencoder2D()
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
    outputs = autoencoder(input_tensor)
    loss = criterion(outputs, input_tensor)
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
torch.save(autoencoder.state_dict(), 'autoencoder2d.pt')

# Save training log to CSV
print("Saving training log to CSV...")
with open('training_log.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Loss'])  # Header
    csv_writer.writerows(training_log)

# Plot the loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('2D Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()  # Close the figure to free memory

# Visualize original and reconstruction
with torch.no_grad():
    reconstructed = autoencoder(input_tensor)

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
    encoded = autoencoder.get_encoded(input_tensor)

# Plot the encoded representation (feature maps)
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axs[i].imshow(encoded[0, i].cpu().detach().numpy(), cmap='viridis')
    axs[i].set_title(f'Feature Map {i+1}')
    axs[i].axis('off')
plt.tight_layout()
plt.savefig('feature_map.png')
plt.close()  # Close the figure to free memory

print("All done! Check the output files.")
