from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, BatchNormalization, \
    MaxPooling2D, UpSampling2D
from keras.models import Model

import tensorflow as tf


def create_model(shape):
    input_img = Input(shape = shape)

    # x = Conv2D(1024, (5, 5), activation = 'sigmoid', padding = 'same')(input_img)
    # decoded = Conv2D(1, (5, 5), activation = 'sigmoid', padding = 'same')(x)

    x = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(input_img)
    decoded = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(x)

    model = Model(input_img, decoded)

    return model


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

    model = create_model(shape = (270, 480, 3))
    model.summary()
    for l in model.layers:
        print(l.name, l.input_shape, l.output_shape)
