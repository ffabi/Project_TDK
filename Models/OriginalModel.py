import os
import sys

import keras.backend as K
import keras.utils.conv_utils as conv_utils
import tensorflow as tf
from keras import applications
from keras.engine.topology import Layer, InputSpec
from keras.layers import Conv2D, LeakyReLU, Concatenate, Dropout
from keras.models import Model, load_model

from Classes.LossGenerator import LossGenerator


class BilinearUpSampling2D(Layer):
    def __init__(self, size = (2, 2), data_format = None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, "size")
        self.input_spec = InputSpec(ndim = 4)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        else:
            assert self.data_format == "channels_last", "Invalid data format: {}".format(self.data_format)
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        if self.data_format == "channels_first":
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        else:
            assert self.data_format == "channels_last", "Invalid data format: {}".format(self.data_format)
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None

        return tf.image.resize_images(inputs, [height, width], method = tf.image.ResizeMethod.BILINEAR,
                                      align_corners = True)

    def get_config(self):
        base_config = super(BilinearUpSampling2D, self).get_config()
        base_config.update({ "size": self.size, "data_format": self.data_format })
        return base_config

# the model has been taken from https://github.com/ialhashim/DenseDepth/blob/master/model.py


def create_model(input_shape, semseg = False):
    print("Loading model: DenseNet169")

    base_model = applications.DenseNet169(
        input_shape = input_shape,
        include_top = False,
        weights = None,
        # dropout_rate = 0.5 # todo update libraries
    )

    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape
    base_model_output_shape = base_model_output_shape.as_list()

    # Layer freezing?
    for layer in base_model.layers:
        layer.trainable = True

    # Starting number of decoder filters
    decode_filters = int(int(base_model_output_shape[-1]) / 2)

    # Define upsampling layer for depth
    def upproject_depth(tensor, filters, name, concat_with):
        up_i = BilinearUpSampling2D((2, 2), name = name + "_upsampling2d")(tensor)
        up_i = Concatenate(name = name + "_concat")(
            [up_i, base_model.get_layer(concat_with).output])  # Skip connection
        up_i = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = "same", name = name + "_convA")(
            up_i)
        # up_i = Dropout(0.5)(up_i)
        up_i = LeakyReLU(alpha = 0.2)(up_i)
        up_i = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = "same", name = name + "_convB")(
            up_i)
        # up_i = Dropout(0.5)(up_i)
        up_i = LeakyReLU(alpha = 0.2)(up_i)
        return up_i

    # Decoder Layers
    decoder = Conv2D(filters = decode_filters, kernel_size = 1, padding = "same",
                     input_shape = base_model_output_shape, name = "conv2")(base_model.output)

    depth_decoder = upproject_depth(decoder, int(decode_filters / 2), "depth_up1", concat_with = "pool3_pool")
    depth_decoder = upproject_depth(depth_decoder, int(decode_filters / 4), "depth_up2", concat_with = "pool2_pool")
    depth_decoder = upproject_depth(depth_decoder, int(decode_filters / 8), "depth_up3", concat_with = "pool1")
    depth_decoder = upproject_depth(depth_decoder, int(decode_filters / 16), "depth_up4", concat_with = "conv1/relu")
    depth_decoder = upproject_depth(depth_decoder, int(decode_filters / 32), "depth_up5", concat_with = "input_1")

    # Extract depths (final layer)
    depth_conv3 = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = "same", name = "depth_output")(depth_decoder)

    if semseg:
        # Define upsampling layer for semantic segmentation
        def upproject_semseg(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name = name + "_upsampling2d")(tensor)
            up_i = Concatenate(name = name + "_concat")(
                [up_i, base_model.get_layer(concat_with).output])  # Skip connection
            up_i = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = "same", name = name + "_convA")(
                up_i)
            # up_i = Dropout(0.5)(up_i)
            up_i = LeakyReLU(alpha = 0.2)(up_i)
            up_i = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = "same", name = name + "_convB")(
                up_i)
            # up_i = Dropout(0.5)(up_i)
            up_i = LeakyReLU(alpha = 0.2)(up_i)
            return up_i

        semseg_decoder = upproject_semseg(decoder, int(decode_filters / 2), "semseg_up1", concat_with = "pool3_pool")
        semseg_decoder = upproject_semseg(semseg_decoder, int(decode_filters / 4), "semseg_up2", concat_with = "pool2_pool")
        semseg_decoder = upproject_semseg(semseg_decoder, int(decode_filters / 8), "semseg_up3", concat_with = "pool1")
        semseg_decoder = upproject_semseg(semseg_decoder, int(decode_filters / 16), "semseg_up4", concat_with = "conv1/relu")
        semseg_decoder = upproject_semseg(semseg_decoder, int(decode_filters / 32), "semseg_up5", concat_with = "input_1")

        # Extract depths (final layer)
        semseg_conv3 = Conv2D(filters = 23, kernel_size = (1,1), strides = 1, padding = "same", name = "semseg_output")(semseg_decoder)

        output = Concatenate(name = "output")([depth_conv3, semseg_conv3])

        # Create the model
        model = Model(inputs = base_model.input, outputs = output)

    else:
        model = Model(inputs = base_model.input, outputs = depth_conv3)

    return model

def load_trained_model(model_file: str) -> Model:
    assert os.path.exists(model_file), "Model not found: {}".format(model_file)

    # todo save and load loss objects as well
    lossgen = LossGenerator(
        mae_w = 1,
        mse_w = 1,
        first_grad_w = 1,
        second_grad_w = 1,
        ssim_w = 1,
        iou_w = 1,
        semseg_w = 1,
    )

    losses = lossgen.get_losses()
    custom_objects = { "BilinearUpSampling2D": BilinearUpSampling2D }

    for loss in losses:
        custom_objects[loss.__name__] = loss


    model = load_model(model_file, custom_objects = custom_objects)
    print("Existing model loaded.")

    return model


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.01)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    with sess.as_default():
        model = create_model((576, 1024, 6), semseg = True)
        model.summary()
    # for l in model.layers[:10]:
    #     print(l.name, l.input_shape, l.output_shape)
