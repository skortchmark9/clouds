import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input, Model
from keras.models import load_model

from data_transforms import load_all_blocks_from_disk
from plotting import plot_block_with_prediction
import matplotlib.pyplot as plt

MAX_VALUE = np.float32(0.021897616)


def weighted_loss(y_true, y_pred):
    """
    Weighted loss function that penalizes errors in irrelevant height levels (z dimension).
    Args:
        y_true: Ground truth tensor of shape (batch_size, i, j, h).
        y_pred: Predicted tensor of shape (batch_size, i, j, h).
    Returns:
        Weighted mean squared error loss.
    """
    # Sum over i, j dimensions to compute total condensation per height level
    truth_altitude_profile = tf.reduce_sum(y_true, axis=(1, 2), keepdims=True)  # Shape: (batch_size, 1, 1, h)

    # Create a weight mask: Penalize layers where the altitude profile is zero
    weight = tf.where(truth_altitude_profile > 0, 1.0, 5.0)  # Adjust penalty factor as needed

    # Compute the levelized loss

    # Compute the Mean Squared Error (MSE) loss
    mse_loss = tf.square(y_true - y_pred)

    # Apply weights to the MSE loss
    weighted_mse = weight * mse_loss

    # Return the average loss across all dimensions
    return tf.reduce_mean(weighted_mse)



def create_model(blocks):
    # Input for top_down data (25x25 grid)
    assert len(blocks[0]['top_down']) == 25
    top_down_input = Input(shape=(25, 25), name='top_down')
    top_down_flattened = layers.Flatten()(top_down_input)
    top_down_dense = layers.Dense(128, activation='relu')(top_down_flattened)

    # Input for altitude profile (1D array of height-level condensation values)
    h_dim = 50
    assert len(blocks[0]['altitude_profile']) == h_dim
    profile_input = Input(shape=(h_dim,), name='altitude_profile')
    profile_input_flattened = layers.Flatten()(profile_input)
    profile_dense = layers.Dense(h_dim, activation='relu')(profile_input_flattened)

    # altitude_profile_softmax = layers.Softmax()(profile_dense)
    # altitude_profile_expanded = layers.Lambda(
    #     lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=2),
    #     output_shape=(1, 1, h_dim)
    # )(altitude_profile_softmax)

    # Combine both features
    print("Before concat")
    print("Shape of top_down_dense:", top_down_dense.shape)
    print("Shape of profile_dense:", profile_dense.shape)
    combined = layers.Concatenate()([top_down_dense, profile_dense])
    print("Combined shape", combined.shape)
    combined_dense = layers.Dense(128, activation='relu')(combined)
    combined_dense = layers.Dense(64, activation='relu')(combined_dense)

    # Output layer matching the shape of 'truth' (flattened), and reshape it
    output = layers.Dense(np.prod(blocks[0]['truth'].shape), activation='relu', name='truth')(combined_dense)
    output_reshaped = layers.Reshape(blocks[0]['truth'].shape)(output)

    # Constrain the output using the altitude profile
    # constrained_output = layers.Multiply()([output_reshaped, altitude_profile_expanded])

    # Define the model
    model = models.Model(inputs={
        'top_down': top_down_input,
        'altitude_profile': profile_input,
    }, outputs=output_reshaped)

    # Compile the model
    model.compile(optimizer='adam', loss=weighted_loss, metrics=['mae'])
    return model


def train_model(model, blocks):
    top_down_data = np.array([block['top_down'] for block in blocks])  # Shape: (num_blocks, 25, 25)
    profile_data = np.array([block['altitude_profile'] for block in blocks])  # Shape: (num_blocks, num_height_levels)
    truth_data = np.array([block['truth'] for block in blocks])  # Shape: (num_blocks, 25, 25, num_height_levels)

    inputs = {
        'top_down': top_down_data,
        'altitude_profile': profile_data
    }
    print('before fit')
    print(inputs['top_down'].shape)
    print(inputs['altitude_profile'].shape)

    # Fit the model on your data
    history = model.fit(
        inputs,
        truth_data,  # target
        epochs=4,  # adjust epochs as needed
        batch_size=20,  # adjust batch size as needed
        validation_split=0.2  # use 20% of data for validation
    )
    return history


def compare_predictions(model: Model, block):
    # Prepare input data based on the structure of `block`
    top_down_input = np.expand_dims(block['top_down'], axis=0)  # Add batch dimension
    altitude_profile_input = np.expand_dims(block['altitude_profile'], axis=0)  # Add batch dimension

    inputs = {
        'top_down': top_down_input,
        'altitude_profile': altitude_profile_input
    }

    # Run model prediction
    prediction = model.predict(inputs)[0]  # Remove batch dimension from prediction

    # Get ground truth data for comparison
    truth = block['truth']

    # Denormalize the data if necessary
    plot_block_with_prediction(truth * MAX_VALUE, prediction * MAX_VALUE)

def histogram_predictions(model: Model, blocks):
    random_blocks = np.random.choice(blocks, 50)

    truth = np.array([block['truth'] for block in random_blocks])

    inputs = {
        'top_down': np.array([block['top_down'] for block in random_blocks]),
        'altitude_profile': np.array([block['altitude_profile'] for block in random_blocks])
    }

    predictions = model.predict(inputs)
    plt.hist(truth.flatten(), bins=10, alpha=0.5, label='Truth')
    plt.hist(predictions.flatten(), bins=10, alpha=0.5, label='Prediction')
    plt.legend()
    plt.title('Distribution of Truth vs Predictions')
    plt.show()
    return predictions



def save_model(model: Model, path='models/model.keras'):
    model.save(path)

def build_and_train_model():
    start = time.time()
    # Load the data
    print("Loading data...")
    blocks = load_all_blocks_from_disk()
    print("Data loaded.")

    # Create the model
    model = create_model(blocks)

    # Train the model
    history = train_model(model, blocks)

    save_model(model)

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return model