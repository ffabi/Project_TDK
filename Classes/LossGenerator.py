import keras.backend as K
import numpy as np
import tensorflow as tf


def mae(y_true, y_pred, weight = 1):
    return K.mean(K.mean(K.abs(y_pred - y_true), axis = -1)) * weight


def mse(y_true, y_pred, weight = 1):
    return K.mean(K.mean(K.square(y_pred - y_true), axis = -1)) * weight


def first_derivative(tensor):
    return tf.image.image_gradients(tensor)


def first_derivative_loss(y_true, y_pred, weight = 1):
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


def second_derivative_loss(y_true, y_pred, weight = 1):
    dy_true, dx_true = second_derivative(y_true)
    dy_pred, dx_pred = second_derivative(y_pred)
    return K.mean(K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis = -1)) * weight


def ssim_loss(y_true, y_pred, weight = 1, max_depth_val = 100):
    # The image sizes must be at least 11x11 because of the filter size.
    return K.mean(K.clip((1 - tf.image.ssim(y_true, y_pred, max_depth_val)) * 0.5, 0, 1)) * weight


def get_absolute_error_image(y_true, y_pred):
    return K.abs(y_pred - y_true)


def get_first_derivative_error_image(y_true, y_pred):
    dy_true, dx_true = first_derivative(y_true)
    dy_pred, dx_pred = first_derivative(y_pred)
    return K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true)


def get_second_derivative_error_image(y_true, y_pred):
    dy_true, dx_true = second_derivative(y_true)
    dy_pred, dx_pred = second_derivative(y_pred)
    return K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true)


class LossGenerator:

    # todo args
    def __init__(self, args = None, mae_w = 1, mse_w = 1, first_grad_w = 1, second_grad_w = 1, ssim_w = 1) -> None:
        super().__init__()

        if args is None:
            self.mae_w = mae_w
            self.mse_w = mse_w
            self.first_grad_w = first_grad_w
            self.second_grad_w = second_grad_w
            self.ssim_w = ssim_w

        else:
            self.mae_w = args.mae_w
            self.mse_w = args.mse_w
            self.first_grad_w = args.first_grad_w
            self.second_grad_w = args.second_grad_w
            self.ssim_w = args.ssim_w

    def set_weights(self, mae_w = 1, mse_w = 1, first_grad_w = 1, second_grad_w = 1, ssim_w = 1) -> None:
        self.mae_w = mae_w
        self.mse_w = mse_w
        self.first_grad_w = first_grad_w
        self.second_grad_w = second_grad_w
        self.ssim_w = ssim_w

    def get_losses(self):
        def combined_weighted_loss(y_true, y_pred):
            return mae(y_true, y_pred, self.mae_w) + \
                   mse(y_true, y_pred, self.mse_w) + \
                   first_derivative_loss(y_true, y_pred, self.first_grad_w) + \
                   second_derivative_loss(y_true, y_pred, self.second_grad_w) + \
                   ssim_loss(y_true, y_pred, self.ssim_w)

        def combined_loss(y_true, y_pred):
            return mae(y_true, y_pred) + \
                   mse(y_true, y_pred) + \
                   first_derivative_loss(y_true, y_pred) + \
                   second_derivative_loss(y_true, y_pred) + \
                   ssim_loss(y_true, y_pred)

        return [mae, mse, first_derivative_loss, second_derivative_loss, ssim_loss, combined_loss,
                combined_weighted_loss]


if __name__ == '__main__':
    # import tensorflow as tf
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.22)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    with sess.as_default():
        # t = tf.constant([[[[1.0], [3 * 2.0], [5 * 3.0]], [[2.0], [3 * 3.0], [5 * 4.0]], [[2 * 3.0], [3 * 4.0], [5 * 5.0]]]])

        t = tf.Variable(tf.random_uniform(shape = (2, 120, 120, 1), dtype = "float32", maxval = 9))
        sess.run(tf.global_variables_initializer())

        # print("t:\n", t[0, :, :, 0].eval())

        # dy = first_derivative(t)[0]
        # print("dy:\n", dy[0, :, :, 0].eval())
        # dx = first_derivative(t)[1]
        # print("dx:\n", dx[0, :, :, 0].eval())

        # dy = tf.image.image_gradients(t)[0]
        # print("dy:\n", dy[0, :, :, 0].eval())
        # dx = tf.image.image_gradients(t)[1]
        # print("dx:\n", dx[0, :, :, 0].eval())

        assert np.equal(first_derivative(t)[0].evaluate(), tf.image.image_gradients(t)[0].evaluate()).all(), \
            "dy is not correct"
        assert np.equal(first_derivative(t)[1].evaluate(), tf.image.image_gradients(t)[1].evaluate()).all(), \
            "dx is not correct"

        # d2y = second_derivative(t)[0]
        # print("d2y:\n", d2y[0, :, :, 0].eval())
        # d2x = second_derivative(t)[1]
        # print("d2x:\n", d2x[0, :, :, 0].eval())

        lossgen = LossGenerator()
        # print(lossgen.loss_generator(t,t))
        for func in lossgen.get_losses():
            print(func.__name__, func(t, t * -1.01).eval())

        # print((ssim_loss(t,t)).dtype)
