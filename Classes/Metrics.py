import numpy as np

# todo use tensors instead


def delta1(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    return (thresh < 1.25).mean()


def delta2(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    return (thresh < 1.25 ** 2).mean()


def delta3(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    return (thresh < 1.25 ** 3).mean()


def avg_rel_err(gt, pred):
    return np.mean(np.abs(gt - pred) / gt)


def rmse(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    return rmse


def log_10(gt, pred):
    pred[pred <= 0] = 0.1
    gt[gt <= 0] = 0.1
    return (np.abs(np.log10(gt) - np.log10(pred))).mean()


def get_eval_metrics():
    return delta1, delta2, delta3, avg_rel_err, rmse, log_10



# import tensorflow as tf
#
#
# def log10(x):
#     numerator = tf.log(x)
#     denominator = tf.log(tf.constant(10, dtype = numerator.dtype))
#     return numerator / denominator
#
#
# def calculate_eval_metrics_tf(gt: tf, pred):
#     thresh = tf.maximum((gt / pred), (pred / gt))
#
#     delta1 = (thresh < 1.25).mean()
#     delta2 = (thresh < 1.25 ** 2).mean()
#     delta3 = (thresh < 1.25 ** 3).mean()
#
#     avg_rel_err = tf.reduce_mean(tf.abs(gt - pred) / gt)
#
#     rmse = tf.math.pow(gt - pred, 2)
#     rmse = tf.sqrt(rmse.mean())
#
#     log_10 = (tf.abs(log10(gt) - log10(pred))).mean()
#
#     return delta1, delta2, delta3, avg_rel_err, rmse, log_10
