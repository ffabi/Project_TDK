import argparse

import tensorflow as tf
from keras.optimizers import Adam

from CallbackGenerator import CallbackGenerator, time
from DataGenerator import DataGenerator
from LossGenerator import LossGenerator
from Models import OriginalModel, simple_dense


def train():
    tf.set_random_seed(1234)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.92)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    # sess = tf.Session()
    input_shape = (576, 1024, 3)
    print("Creating model with input shape:", input_shape)
    model = OriginalModel.create_model(shape = input_shape)
    # model = simple_dense.create_model(shape = input_shape)

    # todo split resized
    # create_resized_dataset(input_shape[0], input_shape[1])

    lossgen = LossGenerator(
        mae_w = 1,
        mse_w = 1,
        first_grad_w = 1,
        second_grad_w = 1,
        ssim_w = 1,
    )

    losses = lossgen.get_losses()
    combined_loss = losses[-1]

    # model.compile(optimizer = Adam(lr = 0.0001, decay = 0), loss = "mse", metrics = ["mse"])
    model.compile(optimizer = Adam(lr = 0.0001, decay = 0), loss = combined_loss, metrics = losses)

    # test_generator = DataGenerator(
    #     set_type = "test",
    #     batch_size = 1,
    #     shuffle = False,
    #     keep_ratio = 0.0026,
    #     shape = input_shape,
    # )

    train_generator = DataGenerator(
        set_type = "train",
        batch_size = 1,
        shuffle = True,
        keep_ratio = 1,
        shape = input_shape,
    )
    val_generator = DataGenerator(
        set_type = "val",
        batch_size = 1,
        shuffle = True,
        keep_ratio = 0.1,
        shape = input_shape,
    )
    assert train_generator.__len__() != 0 and val_generator.__len__() != 0, "No data received from the data generator"

    callback_generator = CallbackGenerator("long_whole_set_201", input_shape)
    callback_generator.set_model(model = model)
    callbacks = callback_generator.get_callbacks()

    with sess.as_default():
        print("Starting training process")
        sess.run(tf.global_variables_initializer())
        model.fit_generator(
            generator = train_generator,
            validation_data = val_generator,
            use_multiprocessing = False,
            shuffle = True,
            epochs = 40,
            max_queue_size = 32,
            callbacks = callbacks,
            # callbacks = None,
        )


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(
        description = 'Monocular Depth Estimation on Synthetic Dataset with Dense Ground Truth')

    # todo
    parser.add_argument('--id', type = int, default = 1, help = 'Batch size')
    parser.add_argument('--batchsize', type = int, default = 1, help = 'Batch size')
    parser.add_argument('--epochs', type = int, default = 100, help = 'Number of epochs')
    parser.add_argument('--eval_freq', type = int, default = 100, help = 'Number of epochs')

    parser.add_argument('--train_keep', type = float, default = 1, help = 'Number of epochs')
    parser.add_argument('--eval_keep', type = float, default = 1, help = 'Number of epochs')
    parser.add_argument('--test_keep', type = float, default = 1, help = 'Number of epochs')

    parser.add_argument('--name', default = 'debug', type = str, help = 'Name prefix')
    parser.add_argument('--checkpoint', type = str, default = '', help = 'Start training from an existing model.')

    parser.add_argument('--shuffle', dest = 'shuffle', action = 'store_true', help = 'Turn shuffle on or off')

    args = parser.parse_args()

    train()
    print((time.time() - start), "s")
