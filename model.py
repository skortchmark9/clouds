import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input, Model

from plotting import plot_3d

def create_model(blocks):
    # Input for top_down data (25x25 grid)
    assert len(blocks[0]['top_down']) == 25
    top_down_input = Input(shape=(25, 25), name='top_down')
    top_down_flattened = layers.Flatten()(top_down_input)
    top_down_dense = layers.Dense(128, activation='relu')(top_down_flattened)

    # Input for altitude profile (1D array of height-level condensation values)
    assert len(blocks[0]['altitude_profile']) == 50
    profile_input = Input(shape=(50,), name='altitude_profile')
    profile_input_flattened = layers.Flatten()(profile_input)
    profile_dense = layers.Dense(64, activation='relu')(profile_input_flattened)

    # Combine both features
    print("Before concat")
    print("Shape of top_down_dense:", top_down_dense.shape)
    print("Shape of profile_dense:", profile_dense.shape)
    combined = layers.Concatenate()([top_down_dense, profile_dense])
    print("Combined shape", combined.shape)
    combined_dense = layers.Dense(128, activation='relu')(combined)
    combined_dense = layers.Dense(64, activation='relu')(combined_dense)
    
    # Output layer matching the shape of 'truth' (flattened), and reshape it
    output = layers.Dense(np.prod(blocks[0]['truth'].shape), activation='linear', name='truth')(combined_dense)
    output_reshaped = layers.Reshape(blocks[0]['truth'].shape)(output)

    # Define the model
    model = models.Model(inputs={
        'top_down': top_down_input,
        'altitude_profile': profile_input,
    }, outputs=output_reshaped)

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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
        epochs=10,  # adjust epochs as needed
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

    plot_3d(prediction)
    plot_3d(truth)
