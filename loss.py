"""
Add customized loss
"""
import tensorflow as tf


def penalized_mean_squared_error(y_true, y_pred, penalty=True):
    y_pred = tf.squeeze(y_pred)
    loss = tf.reduce_mean(tf.losses.mean_squared_error(y_pred, y_true))
    k = tf.cast(y_true.shape[0], 'float32')
    if penalty:
        penalty = tf.square(tf.reduce_sum(tf.abs(y_pred - y_true))) / tf.square(k)
        loss -= penalty
    return loss

def pearson_correlation(y_true, y_pred):
    y_true = tf.constant(y_true)
    denominator = tf.reduce_mean(tf.multiply(y_pred - tf.reduce_mean(y_pred), y_true - tf.reduce_mean(y_true)))
    nominator = tf.sqrt(tf.nn.moments(y_pred, axes=0)[1]) * tf.sqrt(tf.nn.moments(y_true, axes=0)[1])

    return tf.divide(denominator, nominator)

def spearman_correlation(y_true, y_pred):
    k = y_true.shape[0]
    predictions_rank = tf.nn.top_k(y_pred, k=k, sorted=True, name='prediction_rank').indices
    truth_rank = tf.nn.top_k(y_true, k=k, sorted=True, name='real_rank').indices
    return pearson_correlation(y_pred=tf.cast(predictions_rank, 'float32'), y_true=tf.cast(truth_rank, 'float32'))
