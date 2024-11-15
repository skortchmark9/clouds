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


def plot_truth_vs_prediction_3d(truth, prediction):
    """
    Plots the truth and prediction 3D data side by side using `plot_3d`.
    """
    fig = plt.figure(figsize=(20, 7))  # Adjust figure size for side-by-side plots
    
    # Plot Truth
    ax1 = fig.add_subplot(121, projection='3d')
    i_dim, j_dim, h_dim = truth.shape
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    truth_values = truth.flatten()
    i, j, h = i.flatten(), j.flatten(), h.flatten()
    non_zero_mask = truth_values > 0
    ax1.scatter(i[non_zero_mask], j[non_zero_mask], h[non_zero_mask], c=truth_values[non_zero_mask], cmap='viridis', marker='o')
    ax1.set_xlim(0, i_dim - 1)
    ax1.set_ylim(0, j_dim - 1)
    ax1.set_zlim(0, h_dim - 1)
    ax1.set_xlabel("i dimension")
    ax1.set_ylabel("j dimension")
    ax1.set_zlabel("Height dimension (h)")
    ax1.set_title("Truth")
    
    # Plot Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    i_dim, j_dim, h_dim = prediction.shape
    i, j, h = np.meshgrid(np.arange(i_dim), np.arange(j_dim), np.arange(h_dim), indexing='ij')
    pred_values = prediction.flatten()
    i, j, h = i.flatten(), j.flatten(), h.flatten()
    non_zero_mask = pred_values > 0
    scatter = ax2.scatter(i[non_zero_mask], j[non_zero_mask], h[non_zero_mask], c=pred_values[non_zero_mask], cmap='viridis', marker='o')
    ax2.set_xlim(0, i_dim - 1)
    ax2.set_ylim(0, j_dim - 1)
    ax2.set_zlim(0, h_dim - 1)
    ax2.set_xlabel("i dimension")
    ax2.set_ylabel("j dimension")
    ax2.set_zlabel("Height dimension (h)")
    ax2.set_title("Prediction")

    # Add a single colorbar between the two plots
    cbar_ax = fig.add_axes([0.0, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax, label="Condensation Value")

    plt.tight_layout()
    plt.show()
