import keras.backend as K
import numpy as np
import tensorflow as tf


def mae(y_true, y_pred, weight = 1, use_semseg_weight = False):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    return K.mean(K.mean(K.abs(y_pred - y_true), axis = -1)) * weight


def mse(y_true, y_pred, weight = 1, semseg = True):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    if semseg:
        weight_mask = get_semseg_weight(y_true)
        return K.mean(K.mean(K.square(y_pred - y_true) * weight_mask, axis = -1)) * weight
    else:
        return K.mean(K.mean(K.square(y_pred - y_true), axis = -1)) * weight


def get_semseg_weight(semseg_23):
    ped_mask = semseg_23[..., 12] == 1
    car_mask = semseg_23[..., 14] == 1
    base = semseg_23[..., 0] != 42

    weight_mask = tf.cast(ped_mask, tf.float32) * 99 + tf.cast(car_mask, tf.float32) * 99 + tf.cast(base, tf.float32)

    return weight_mask


def first_derivative(tensor):
    return tf.image.image_gradients(tensor)


def first_derivative_loss(y_true, y_pred, weight = 1, use_semseg_weight = False):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    dy_true, dx_true = first_derivative(y_true)
    dy_pred, dx_pred = first_derivative(y_pred)
    return K.mean(K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis = -1)) * weight


def second_derivative(tensor):
    # todo check out Hesse matrix and this:

    # def second_derivative(x):
    #     assert x.dim() == 4, "expected 4-dimensional data, but instead got {}".format(x.dim())
    #     horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :-2] - x[:, :, 1:-1, 2:]
    #     vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:-1] - x[:, :, 2:, 1:-1]
    #     der_2nd = horizontal.abs() + vertical.abs()
    #     return der_2nd.mean()
    #

    dy, dx = tf.image.image_gradients(tensor)

    d2y, _ = tf.image.image_gradients(dy[:, :-1, :, :])
    _, d2x = tf.image.image_gradients(dx[:, :, :-1, :])

    batch_size, height, width, depth = tf.unstack(tf.shape(tensor))

    shape = tf.stack([batch_size, 1, width, depth])
    d2y = tf.concat([d2y, tf.zeros(shape, tensor.dtype)], 1)
    d2y = tf.reshape(d2y, tf.shape(tensor))

    shape = tf.stack([batch_size, height, 1, depth])
    d2x = tf.concat([d2x, tf.zeros(shape, tensor.dtype)], 2)
    d2x = tf.reshape(d2x, tf.shape(tensor))

    return d2y, d2x


def second_derivative_loss(y_true, y_pred, weight = 1, use_semseg_weight = False):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    dy_true, dx_true = second_derivative(y_true)
    dy_pred, dx_pred = second_derivative(y_pred)
    return K.mean(K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis = -1)) * weight


def ssim_loss(y_true, y_pred, weight = 1, max_depth_val = 100, use_semseg_weight = False):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    # The image sizes must be at least 11x11 because of the filter size.
    return K.mean(K.clip((1 - tf.image.ssim(y_true, y_pred, max_depth_val)) * 0.5, 0, 1)) * weight


def get_absolute_error_image(y_true, y_pred):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    return K.abs(y_pred - y_true)


def get_first_derivative_error_image(y_true, y_pred):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    dy_true, dx_true = first_derivative(y_true)
    dy_pred, dx_pred = first_derivative(y_pred)
    return K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true)


def get_second_derivative_error_image(y_true, y_pred):
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]

    dy_true, dx_true = second_derivative(y_true)
    dy_pred, dx_pred = second_derivative(y_pred)
    return K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true)


# semseg loss
def iou_loss(y_true, y_pred, weight = 1):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    return (intersection + 1) / (union + 1) * weight


# def semseg_acc_loss(y_true, y_pred, weight = 1):
#     y_true = y_true[..., 1:]
#     y_pred = y_pred[..., 1:]
#
#     correct = tf.math.equal(y_pred, y_true)
#     correct = K.mean(tf.cast(correct, tf.float32))
#     return correct * weight

def semseg_crossentropy(y_true, y_pred, weight = 1):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_true, labels = y_pred))
    return loss * weight


class LossGenerator:

    def __init__(self, args = None, mae_w = 1, mse_w = 1, first_grad_w = 1, second_grad_w = 1, ssim_w = 1, iou_w = 1):
        super().__init__()

        if args is None:
            self.mae_w = mae_w
            self.mse_w = mse_w
            self.first_grad_w = first_grad_w
            self.second_grad_w = second_grad_w
            self.ssim_w = ssim_w
            self.iou_w = iou_w

        else:
            self.mae_w = args.mae_w
            self.mse_w = args.mse_w
            self.first_grad_w = args.first_grad_w
            self.second_grad_w = args.second_grad_w
            self.ssim_w = args.ssim_w
            self.iou_w = args.iou_w

    def get_losses(self):
        def combined_weighted_loss(y_true, y_pred):
            return mae(y_true, y_pred, self.mae_w) + \
                   mse(y_true, y_pred, self.mse_w) + \
                   first_derivative_loss(y_true, y_pred, self.first_grad_w) + \
                   second_derivative_loss(y_true, y_pred, self.second_grad_w) + \
                   ssim_loss(y_true, y_pred, self.ssim_w) \
                   + iou_loss(y_true, y_pred, self.iou_w)

        def combined_loss(y_true, y_pred):
            return mae(y_true, y_pred) + \
                   mse(y_true, y_pred) + \
                   first_derivative_loss(y_true, y_pred) + \
                   second_derivative_loss(y_true, y_pred) + \
                   ssim_loss(y_true, y_pred) \
                   + iou_loss(y_true, y_pred, self.iou_w)

        return [mae, mse, first_derivative_loss, second_derivative_loss, ssim_loss, combined_loss, iou_loss,
                combined_weighted_loss]
