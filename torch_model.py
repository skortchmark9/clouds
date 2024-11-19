import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_transforms import load_all_blocks_from_disk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random


# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder: Combine both inputs
        self.encoder_fc1 = nn.Linear(25*25 + 50*1, 512)
        self.encoder_fc2 = nn.Linear(512, 128)
        self.fc_mu = nn.Linear(128, 64)
        self.fc_logvar = nn.Linear(128, 64)

        # Decoder: Decode latent space to output
        self.decoder_fc1 = nn.Linear(64, 128)
        self.decoder_fc2 = nn.Linear(128, 512)
        self.decoder_fc3 = nn.Linear(512, 25*25*50)

    def encode(self, x):
        h1 = F.relu(self.encoder_fc1(x))
        h2 = F.relu(self.encoder_fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.decoder_fc1(z))
        h4 = F.relu(self.decoder_fc2(h3))
        h5 = torch.sigmoid(self.decoder_fc3(h4))
        return h5.view(-1, 25, 25, 50)

    def forward(self, top_down, side_view):
        # Flatten inputs and concatenate
        top_down_flat = top_down.view(top_down.size(0), -1)
        side_view_flat = side_view.view(side_view.size(0), -1)
        x = torch.cat([top_down_flat, side_view_flat], dim=1)

        # Encode to latent space
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Decode to output shape
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Define loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

def create_and_train_model(blocks):
    # Generate random input and output data for testing
    top_down_input = torch.tensor([block['top_down'] for block in blocks])  # Shape: (num_blocks, 25, 25)
    side_view_input = torch.tensor([block['altitude_profile'] for block in blocks])  # Shape: (num_blocks, num_height_levels)
    output_data = torch.tensor([block['truth'] for block in blocks])  # Shape: (num_blocks, 25, 25, num_height_levels)

    batch_size = 64
    # top_down_input = torch.rand(batch_size, 25, 25)  # Shape [25, 25]
    # side_view_input = torch.rand(batch_size, 50, 1)  # Shape [50, 1]
    # output_data = torch.rand(batch_size, 25, 25, 50)  # Shape [25, 25, 50]

    # Dataset and DataLoader
    dataset = TensorDataset(top_down_input, side_view_input, output_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Initialize the model, optimizer, and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    print(list(model.named_parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for top_down, side_view, target in dataloader:
            top_down, side_view, target = top_down.to(device), side_view.to(device), target.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(top_down, side_view)
            loss = vae_loss(recon_x, target, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(dataloader.dataset):.4f}")

    return model

def plot_block_with_prediction(model, block):
    """
    Plots the inputs and 3D truth/prediction for a given block.
    Args:
        model: The trained model.
        block: A dictionary containing the block's data (truth, top_down, altitude_profile).
    """
    # Extract inputs and truth from the block
    top_down = torch.tensor(block['top_down']).unsqueeze(0)
    altitude_profile = torch.tensor(block['altitude_profile']).unsqueeze(1).unsqueeze(0)
    truth = block['truth']
    print(top_down.shape, altitude_profile.shape, truth.shape)

    prediction, _, _ = model(top_down, altitude_profile)

    initial_prediction = prediction.detach().numpy()
    prediction = initial_prediction[0]
    print(prediction.shape)
    # print(block['time'])

    print(prediction)

    print('truth', np.sum(truth), 'pred', np.sum(prediction))

    top_down = top_down.squeeze().numpy()
    altitude_profile = altitude_profile.squeeze().numpy()


    # Prepare the figure
    fig = plt.figure(figsize=(12, 8))  # Smaller figure size
    # fig = plt.figure(figsize=(16, 12))  # Adjust figure size

    # Plot the top-down input
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(top_down, cmap='viridis')
    ax1.set_title("Top-Down Input")
    fig.colorbar(im1, ax=ax1)

    # Plot the altitude profile input
    ax2 = fig.add_subplot(222)
    levels = np.arange(len(altitude_profile))
    ax2.plot(altitude_profile, levels, label="Altitude Profile")
    # ax2.invert_yaxis()  # Invert for altitude display
    ax2.set_xlabel("Condensation Value")
    ax2.set_ylabel("Height Levels")
    ax2.set_title("Altitude Profile Input")
    ax2.legend()

    # Plot the truth 3D grid
    ax3 = fig.add_subplot(223, projection='3d')
    i_dim, j_dim, h_dim = truth.shape
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    truth_values = truth.flatten()
    i, j, h = i.flatten(), j.flatten(), h.flatten()
    non_zero_mask = truth_values > 0
    ax3.scatter(i[non_zero_mask], j[non_zero_mask], h[non_zero_mask], c=truth_values[non_zero_mask], cmap='viridis', marker='o')
    ax3.set_xlim(0, i_dim - 1)
    ax3.set_ylim(0, j_dim - 1)
    ax3.set_zlim(0, h_dim - 1)
    ax3.set_xlabel("i dimension")
    ax3.set_ylabel("j dimension")
    ax3.set_zlabel("Height dimension (h)")
    ax3.set_title("3D Truth")

    # Plot the predicted 3D grid
    ax4 = fig.add_subplot(224, projection='3d')
    i_dim, j_dim, h_dim = prediction.shape
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    i = i.flatten()
    j = j.flatten()
    h = h.flatten()
    pred_values = prediction.flatten()
    non_zero_mask = pred_values > 0
    ax4.scatter(i[non_zero_mask], j[non_zero_mask], h[non_zero_mask], c=pred_values[non_zero_mask], cmap='viridis', marker='o')
    ax4.set_xlim(0, i_dim - 1)
    ax4.set_ylim(0, j_dim - 1)
    ax4.set_zlim(0, h_dim - 1)
    ax4.set_xlabel("i dimension")
    ax4.set_ylabel("j dimension")
    ax4.set_zlabel("Height dimension (h)")
    ax4.set_title("3D Prediction")
    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

