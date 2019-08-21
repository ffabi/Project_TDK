from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, BatchNormalization, \
    MaxPooling2D, UpSampling2D
from keras.models import Model

import tensorflow as tf


def create_model(shape):
    input_img = Input(shape = shape)

    scale = 1

    x = Conv2D(256 * scale, (3, 3), activation = 'relu', padding = 'same')(input_img)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(128 * scale, (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(64 * scale, (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(32 * scale, (3, 3), activation = 'relu', padding = 'same')(x)
    encoded = MaxPooling2D((2, 2), padding = 'same')(x)

    x = Conv2D(32 * scale, (3, 3), activation = 'relu', padding = 'same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64 * scale, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128 * scale, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256 * scale, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(x)

    autoencoder = Model(input_img, decoded)

    return autoencoder


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

    model = create_model((720, 1280, 3))
    model.summary()
    for l in model.layers:
        print(l.name, l.input_shape, l.output_shape)
