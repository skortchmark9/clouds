import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def plot_truth(block):
    # Example data for demonstration
    # Replace `block.truth` with your actual data if available
    # Assuming block.truth has shape (25, 25, 50)
    truth = block['truth']  # Replace with block.truth
    i_dim, j_dim, h_dim = truth.shape

    # Prepare the data for plotting
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    values = truth.flatten()  # Flatten the truth array for color mapping

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

    plt.title("3D Plot of `block.truth` Values")
    plt.show()

