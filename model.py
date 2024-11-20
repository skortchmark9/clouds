import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input, Model
from keras.models import load_model



from data_transforms import load_all_blocks_from_disk
from plotting import plot_block_with_prediction
import matplotlib.pyplot as plt
from loss import weighted_loss_with_layerwise_sum_constraint, layerwise_sum_error

def ssim_loss(y_true, y_pred):
    """
    Compute SSIM loss between the true and predicted values.
    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
    Returns:
        SSIM loss (1 - SSIM index), averaged across the batch.
    """
    ssim_index = tf.image.ssim(y_true, y_pred, max_val=1.0)  # SSIM index in range [0, 1]
    return 1.0 - tf.reduce_mean(ssim_index)  # Loss = 1 - SSIM



MAX_VALUE = np.float32(0.021897616)

def multiply_model_with_height_embeddings(blocks):
    # Get height dimension from example block
    h_dim = len(blocks[0]['altitude_profile'])

    # Input for top-down data (25x25 grid)
    top_down_input = Input(shape=(25, 25), name='top_down')

    # Input for altitude profile (1D array of height-level condensation values)
    profile_input = Input(shape=(h_dim,), name='altitude_profile')

   # Trainable embedding for height levels
    n_embeddings = 32
    height_indices = tf.range(h_dim, dtype=tf.int32)  # Indices for height levels (0 to h_dim-1)
    height_embedding_layer = layers.Embedding(input_dim=h_dim, output_dim=n_embeddings, name='height_embedding')
    height_embeddings = height_embedding_layer(height_indices)  # Shape: (h_dim, n_embeddings)

    # Expand altitude profile for element-wise multiplication
    profile_expanded = layers.Lambda(
        lambda x: tf.expand_dims(x, axis=-1),
        name='expand_profile'
    )(profile_input)  # Shape: (batch_size, h_dim, 1)

    print('height embeddings shape', height_embeddings.shape)
    print('altitude profile expanded shape', profile_expanded.shape)
    # Tile height embeddings to match the batch size
    height_embeddings_batch = layers.Lambda(
        lambda inputs: tf.tile(
            tf.expand_dims(inputs[0], axis=0),  # Add batch dimension to height_embeddings
            [tf.shape(inputs[1])[0], 1, 1]  # Tile to match batch size
        ),
        name='tile_height_embeddings'
    )([height_embeddings, profile_input])  # Pass height_embeddings and profile_input

    print('height embeddings batch', height_embeddings_batch.shape)

    # Element-wise multiplication of altitude profile and embeddings
    altitude_profile_with_embeddings = layers.Multiply(
        name='multiply_profile_and_embeddings'
    )([profile_expanded, height_embeddings_batch])  # Shape: (batch_size, h_dim, n_embeddings)
    
    # Reduce dimensionality using a deeper network
    altitude_profile_reduced = layers.Dense(
        n_embeddings, activation='relu', name='dense1'
    )(altitude_profile_with_embeddings)  # Shape: (batch_size, h_dim, n_embeddings)
    altitude_profile_reduced = layers.Dense(
        1, activation=None, name='dense2'
    )(altitude_profile_reduced)  # Shape: (batch_size, h_dim, 1)

    # Squeeze the reduced dimension
    altitude_profile_squeezed = layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1),
        name='squeeze_reduced_profile'
    )(altitude_profile_reduced)  # Shape: (h_dim,)

    altitude_profile_broadcasted = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, axis=1), axis=2), [1, 25, 25, 1]),
        output_shape=(25, 25, h_dim),
        name='altitude_profile_broadcasted'
    )(altitude_profile_squeezed)
    print("alt profile broadcasted", altitude_profile_broadcasted.shape)

    # Multiply altitude profile with top-down input
    top_down_expanded = layers.Lambda(
        lambda x: tf.expand_dims(x, axis=-1),
        name='top_down_expanded'
    )(top_down_input)
    output = layers.Multiply()([top_down_expanded, altitude_profile_broadcasted])
    print("output", output.shape)

    # Renormalize so that the sum of the output matches the sum of the top-down input
    output_sum = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True),
        output_shape=(1,),
        name='output_sum',
    )(output)

    # Compute the sum of the top-down input
    top_down_sum = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True),
        output_shape=(1,),
        name='top_down_sum',
    )(top_down_expanded)

    # Renormalize the output
    output_renormalized = layers.Lambda(
        lambda x: x[0] / (x[1] + 1e-8) * x[2],
        output_shape=(25, 25, h_dim),
        name='output_renormalized',
    )([output, output_sum, top_down_sum])



    # Define the model
    model = models.Model(inputs={
        'top_down': top_down_input,
        'altitude_profile': profile_input,
    }, outputs=output_renormalized)

    # Compile the model (no training necessary for this test)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model



def simple_multiply_model(blocks):
    # Get height dimension from example block
    h_dim = len(blocks[0]['altitude_profile'])

    # Input for top-down data (25x25 grid)
    top_down_input = Input(shape=(25, 25), name='top_down')

    # Input for altitude profile (1D array of height-level condensation values)
    profile_input = Input(shape=(h_dim,), name='altitude_profile')

    altitude_profile_broadcasted = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, axis=1), axis=2), [1, 25, 25, 1]),
        output_shape=(25, 25, h_dim),
        name='altitude_profile_broadcasted'
    )(profile_input)
    print("alt profile broadcasted", altitude_profile_broadcasted.shape)

    # Multiply altitude profile with top-down input
    top_down_expanded = layers.Lambda(
        lambda x: tf.expand_dims(x, axis=-1),
        name='top_down_expanded'
    )(top_down_input)


    output_mult = layers.Multiply()([top_down_expanded, altitude_profile_broadcasted])
    print("output_mult", output_mult.shape)

    # Expand output_mult to add a singleton channel dimension
    output_mult_expanded = layers.Lambda(
        lambda x: tf.expand_dims(x, axis=-1), name='expand_output_mult'
    )(output_mult)  # Shape: (batch_size, 50, 25, 25, 1)

    # Apply Conv3D for feature extraction
    conv3d_output = layers.Conv3D(
        filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv3d_feature_extraction',
    )(output_mult_expanded)  # Shape: (batch_size, 48, 23, 23, 16)

    # Reduce to a single channel for condensate levels
    conv3d_single_channel = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv3d_output'
    )(conv3d_output)  # Shape: (batch_size, 50, 25, 25, 1)
    print(conv3d_single_channel.shape)

    # Crop walls
    # conv3d_cropped = layers.Cropping3D(
    #     cropping=((1, 1), (1, 1), (1, 1)), name="conv3d_cropped"
    # )(conv3d_single_channel)  # Shape: (batch_size, 23, 23, 48)

    # Pad the second Conv3D layer output to match `output_mult`
    conv3d_uncropped = layers.ZeroPadding3D(
        padding=((1, 1), (1, 1), (1, 1)), name="conv3d_uncropped"
    )(conv3d_single_channel)  # Shape: (batch_size, 50, 25, 25, 1)


    conv_output = layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name='squeeze_final_output'
    )(conv3d_single_channel)  # Shape: (batch_size, 50, 25, 25)

    # Compute the sum of the top-down input
    top_down_sum = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True),
        output_shape=(1,),
        name='top_down_sum',
    )(top_down_expanded)

    # Renormalize so that the sum of the output matches the sum of the top-down input
    conv_output_sum = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True),
        output_shape=(1,),
        name='conv_output_sum',
    )(conv_output)


    # Renormalize the output
    conv_output_renormalized = layers.Lambda(
        lambda x: x[0] / (x[1] + 1e-10) * x[2],
        output_shape=(25, 25, h_dim),
        name='conv_output_renormalized',
    )([conv_output, conv_output_sum, top_down_sum])



    output = layers.Add()([
        layers.Lambda(lambda x: x * 1.0, name='conv_3d_weight')(conv_output_renormalized),
        layers.Lambda(lambda x: x * 0.0, name='output_mult_weight')(output_mult)
    ])


    # Renormalize so that the sum of the output matches the sum of the top-down input
    output_sum = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True),
        output_shape=(1,),
        name='output_sum',
    )(output)

    # Renormalize the output
    output_renormalized = layers.Lambda(
        lambda x: x[0] / (x[1] + 1e-10) * x[2],
        output_shape=(25, 25, h_dim),
        name='output_renormalized',
    )([output, output_sum, top_down_sum])



    # Define the model
    model = models.Model(inputs={
        'top_down': top_down_input,
        'altitude_profile': profile_input,
    }, outputs=output)

    model.compile(optimizer='adam', loss=ssim_loss, metrics=['mae'])

    return model

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
    model.compile(optimizer='adam', loss=[weighted_loss_with_layerwise_sum_constraint], metrics=['mae', 'mse', layerwise_sum_error])
    return model


def train_model(model, blocks, epochs=10):
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
        epochs=epochs,  # adjust epochs as needed
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


def show_model_architecture(model: Model):
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, to_file="model.png")

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