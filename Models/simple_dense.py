import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model


def create_model(shape):
    input_img = Input(shape = shape)

    # x = Conv2D(1024, (5, 5), activation = 'sigmoid', padding = 'same')(input_img)
    # decoded = Conv2D(1, (5, 5), activation = 'sigmoid', padding = 'same')(x)

    x = Dense(1000)(input_img)
    decoded = Dense(1)(x)

    mlp = Model(input_img, decoded)

    return mlp


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

    model = create_model(shape = (576, 1024, 3))
    model.summary()
    for l in model.layers:
        print(l.name, l.input_shape, l.output_shape)
