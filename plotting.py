import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

def plot_top_down(block):
    data = block['top_down']
    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Top Down Values')
    plt.title("Top Down Data (Example)")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()

def plot_altitude_profile(block):
    # Plot a single example from altitude_profile
    plt.plot(block['altitude_profile'], range(len(block['altitude_profile'])))
    plt.title("Altitude Profile (Example)")
    plt.xlabel("Condensation Values")
    plt.ylabel("Height Levels")
    plt.show()

def plot_3d(block_3d, title='Condensation'):
    # Example data for demonstration
    # Replace `block.truth` with your actual data if available
    # Assuming block.truth has shape (25, 25, 50)
    i_dim, j_dim, h_dim = block_3d.shape

    # Prepare the data for plotting
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    values = block_3d.flatten()  # Flatten the truth array for color mapping

    # Flatten the coordinates for 3D scatter plot
    i = i.flatten()
    j = j.flatten()
    h = h.flatten()

    # Mask to keep only non-zero values
    non_zero_mask = values != 0
    i = i[non_zero_mask]
    j = j[non_zero_mask]
    h = h[non_zero_mask]
    values = values[non_zero_mask]

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Set full range for each axis
    ax.set_xlim(0, i_dim - 1)
    ax.set_ylim(0, j_dim - 1)
    ax.set_zlim(0, h_dim - 1)

    # Plot the data with color intensity representing the values in `truth`
    scatter = ax.scatter(i, j, h, c=values, cmap='viridis', marker='o')
    fig.colorbar(scatter, ax=ax, label='Condensation Value')

    # Set labels
    ax.set_xlabel("i dimension")
    ax.set_ylabel("j dimension")
    ax.set_zlabel("Height dimension (h)")

    plt.title(f"3D Plot of {title} Values")
    plt.show()


def pick(blocks):
    idx = random.randint(0, len(blocks) - 1)
    print('block idx', idx)
    return blocks[idx]

def plot_block_with_prediction(model, blocks, idx=None):
    """
    Plots the inputs and 3D truth/prediction for a given block.
    Args:
        model: The trained model.
        block: A dictionary containing the block's data (truth, top_down, altitude_profile).
    """
    if idx is not None:
        block = blocks[idx]
    else:
        idx = random.randint(0, len(blocks) - 1)

    block = blocks[idx]


    # Extract inputs and truth from the block
    top_down = block['top_down']
    altitude_profile = block['altitude_profile']
    truth = block['truth']

    # Prepare inputs for the model
    inputs = {
        'top_down': np.expand_dims(top_down, axis=0),
        'altitude_profile': np.expand_dims(altitude_profile, axis=0)
    }
    # Predict the 3D grid
    prediction = model.predict(inputs)[0]  # Remove batch dimension
    print(prediction.shape)
    print(block['time'])
    print('truth', np.sum(truth), 'pred', np.sum(prediction))


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

    # Calculate global color limits
    vmin = min(truth.min(), prediction.min())
    vmax = max(truth.max(), prediction.max())

    # Plot the truth 3D grid
    ax3 = fig.add_subplot(223, projection='3d')
    i_dim, j_dim, h_dim = truth.shape
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    truth_values = truth.flatten()
    i, j, h = i.flatten(), j.flatten(), h.flatten()
    non_zero_mask = truth_values > 0
    ax3.scatter(i[non_zero_mask], j[non_zero_mask], h[non_zero_mask], c=truth_values[non_zero_mask], cmap='viridis', marker='o', vmin=vmin, vmax=vmax)
    ax3.set_xlim(0, i_dim - 1)
    ax3.set_ylim(0, j_dim - 1)
    ax3.set_zlim(0, h_dim - 1)
    ax3.set_xlabel("i dimension")
    ax3.set_ylabel("j dimension")
    ax3.set_zlabel("Height dimension (h)")
    ax3.set_title(f"3D Truth of Block {idx}")

    # Plot the predicted 3D grid
    ax4 = fig.add_subplot(224, projection='3d')
    i_dim, j_dim, h_dim = prediction.shape
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    i = i.flatten()
    j = j.flatten()
    h = h.flatten()
    pred_values = prediction.flatten()
    non_zero_mask = pred_values > 0
    ax4.scatter(i[non_zero_mask], j[non_zero_mask], h[non_zero_mask], c=pred_values[non_zero_mask], cmap='viridis', marker='o', vmin=vmin, vmax=vmax)
    ax4.set_xlim(0, i_dim - 1)
    ax4.set_ylim(0, j_dim - 1)
    ax4.set_zlim(0, h_dim - 1)
    ax4.set_xlabel("i dimension")
    ax4.set_ylabel("j dimension")
    ax4.set_zlabel("Height dimension (h)")
    ax4.set_title(f"3D Prediction of Block {idx}")
    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

