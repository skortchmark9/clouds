import tensorflow as tf


def layerwise_sum_error(y_true, y_pred):
    sum_true_layerwise = tf.reduce_sum(y_true, axis=[1, 2])  # Shape: (batch_size, h)
    sum_pred_layerwise = tf.reduce_sum(y_pred, axis=[1, 2])  # Shape: (batch_size, h)
    return tf.reduce_mean(tf.abs(sum_true_layerwise - sum_pred_layerwise))  # Mean absolute error


def weighted_loss_with_layerwise_sum_constraint(y_true, y_pred):
    # Weighted MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    layer_error = 2 * layerwise_sum_error(y_true, y_pred)

    field__sum_error = 1 * tf.reduce_mean(tf.abs(tf.reduce_sum(y_true, axis=[1, 2]) - tf.reduce_sum(y_pred, axis=[1, 2])))

    # Combine the two losses with a weighting factor
    total_loss = mse_loss + field__sum_error + layer_error

    return total_loss




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
