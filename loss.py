"""
Add customized loss or metrics
"""
import tensorflow as tf
from functools import partial


# def penalized_mean_squared_error(y_true, y_pred, penalty=True):
#     y_pred = tf.squeeze(y_pred)
#     loss = tf.reduce_mean(tf.losses.mean_squared_error(y_pred, y_true))
#     k = tf.cast(y_true.shape[0], 'float32')
#     if penalty:
#         penalty = tf.square(tf.reduce_sum(tf.abs(y_pred - y_true))) / tf.square(k)
#         loss -= penalty
#     return loss

def penalized_mean_squared_error(y_true, y_pred, penalty=True):
    # y_pred = tf.squeeze(y_pred)
    def _get_masked_mse(elems):
        y_true, y_pred = elems
        penalty = True
        mask = tf.greater_equal(y_true, 0.)
        k = tf.reduce_sum(tf.cast(mask, dtype=tf.float32))
        truth = tf.squeeze(tf.boolean_mask(tensor=y_true, mask=mask))
        truth = tf.cast(truth, dtype=tf.float32)
        pred = tf.squeeze(tf.boolean_mask(tensor=y_pred, mask=mask))
        pred = tf.cast(pred, dtype=tf.float32)
        loss = tf.losses.mean_squared_error(truth, pred)
        if penalty:
            penalty = tf.reduce_sum(tf.square(truth - pred)) / tf.square(k)
            loss -= penalty
        return loss

    return tf.reduce_sum(tf.map_fn(lambda x: _get_masked_mse(x), elems=(tf.transpose(y_true), tf.transpose(y_pred)), dtype=tf.float32))


def pearson_correlation(y_true, y_pred):
    def _get_corr(elems):
        y_true, y_pred = elems
        mask = tf.greater_equal(y_true, 0.)
        truth = tf.squeeze(tf.boolean_mask(tensor=y_true, mask=mask))
        truth = tf.cast(truth, dtype=tf.float32)
        pred = tf.squeeze(tf.boolean_mask(tensor=y_pred, mask=mask))
        pred = tf.cast(pred, dtype=tf.float32)
        #y_true = tf.constant(y_true)
        denominator = tf.reduce_mean(tf.multiply(truth - tf.reduce_mean(truth), pred - tf.reduce_mean(pred)))
        nominator = tf.sqrt(tf.nn.moments(truth, axes=0)[1]) * tf.sqrt(tf.nn.moments(pred, axes=0)[1])
        return tf.math.divide_no_nan(denominator, nominator)
    return tf.map_fn(lambda x: _get_corr(x), elems=(tf.transpose(y_true), tf.transpose(y_pred)), dtype=tf.float32)


def p_corr_helper(y_true, y_pred):
    y_true = tf.constant(y_true)
    denominator = tf.reduce_mean(tf.multiply(y_pred - tf.reduce_mean(y_pred), y_true - tf.reduce_mean(y_true)))
    nominator = tf.sqrt(tf.nn.moments(y_pred, axes=0)[1]) * tf.sqrt(tf.nn.moments(y_true, axes=0)[1])
    return tf.math.divide_no_nan(denominator, nominator)


def spearman_correlation(y_true, y_pred):
    def _get_corr(elems):
        y_true, y_pred = elems
        mask = tf.greater_equal(y_true, 0.)
        truth = tf.squeeze(tf.boolean_mask(tensor=y_true, mask=mask))
        truth = tf.cast(truth, dtype=tf.float32)
        pred = tf.squeeze(tf.boolean_mask(tensor=y_pred, mask=mask))
        pred = tf.cast(pred, dtype=tf.float32)
        pred_rank = tf.argsort(tf.argsort(pred))
        truth_rank = tf.argsort(tf.argsort(truth))
        return p_corr_helper(y_true=tf.cast(truth_rank, 'float32'), y_pred=tf.cast(pred_rank, 'float32'))
    return tf.map_fn(lambda x: _get_corr(x), elems=(tf.transpose(y_true), tf.transpose(y_pred)), dtype=tf.float32)

def compute_cosine_distances_matrix(x, y):
    normalize_x = tf.nn.l2_normalize(x, 1)
    normalize_y = tf.nn.l2_normalize(y, 1)
    sim_matrix = tf.matmul(normalize_x, normalize_y, transpose_b=True)
    return sim_matrix


def contrastive_loss(y_true, y_pred):
    sim_matrix = compute_cosine_distances_matrix(y_true, y_pred)
    return tf.math.log(tf.reduce_sum(tf.multiply(tf.exp(sim_matrix), -2 * (
            tf.eye(num_rows=sim_matrix.shape[0], dtype=tf.float32) - 1)))) - tf.math.log(
        tf.reduce_sum(tf.multiply(tf.exp(sim_matrix), tf.eye(num_rows=sim_matrix.shape[0], dtype=tf.float32))))


def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0)
    return cost


def mmd_loss(y_true, y_pred):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = maximum_mean_discrepancy(
        y_true, y_pred, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value)

    return loss_value
