import argparse
from datetime import datetime

from keras.optimizers import Adam

from Models import OriginalModel
from Classes.CustomCallback import *


def train(args):
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
    # model.compile(optimizer = RAdam(), loss = combined_loss, metrics = losses)

    train_generator = DataGenerator(
        args = args,
        set_type = "train",
        batch_size = args.batchsize,
        shuffle = not args.no_shuffle,
        keep_ratio = args.train_keep,
        shape = input_shape,
    )
    val_generator = DataGenerator(
        args = args,
        set_type = "val",
        batch_size = 1,
        shuffle = not args.no_shuffle,
        keep_ratio = args.val_keep,
        shape = input_shape,
    )
    assert train_generator.__len__() != 0 and val_generator.__len__() != 0, "No data received from the data generator"

    name = args.name + "_" + str(args.id) + "_" + datetime.now().strftime("%m_%d_%H_%M")

    callbacks = get_callbacks(args, os.path.join("../results", name))

    with sess.as_default():
        print("Starting training process")
        sess.run(tf.global_variables_initializer())
        model.fit_generator(
            generator = train_generator,
            validation_data = val_generator,
            use_multiprocessing = False,
            shuffle = not args.no_shuffle,
            epochs = args.epochs,
            max_queue_size = 32,
            callbacks = callbacks,
            # callbacks = None,
        )


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(
        description = 'Monocular Depth Estimation on Synthetic Dataset with Dense Ground Truth')

    parser.add_argument('--id', type = int, default = 0, help = 'Train id')
    parser.add_argument('--batchsize', type = int, default = 1, help = 'Batch size')
    parser.add_argument('--epochs', type = int, default = 42, help = 'Number of epochs')
    parser.add_argument('--eval_freq', type = int, default = 100, help = 'todo')

    parser.add_argument('--train_keep', type = float, default = 1, help = '-')
    parser.add_argument('--val_keep', type = float, default = 1, help = '-')
    parser.add_argument('--test_keep', type = float, default = 1, help = '-')

    parser.add_argument('--name', default = 'debug', type = str, help = 'Name prefix')
    parser.add_argument('--checkpoint', type = str, default = '', help = 'Start training from an existing model.')
    parser.add_argument('--shape', type = str, default = '(576, 1024)', help = 'Start training from an existing model.')

    # todo default false
    parser.add_argument('--no_shuffle', dest = 'no_shuffle', action = 'store_true', help = 'Turn shuffle off')
    parser.add_argument('--overfit', dest = 'overfit', action = 'store_true', help = 'Overfit on test set')

    args = parser.parse_args()
    
    args.shape = eval(args.shape)

    print(vars(args))
    train(args)
    print((time.time() - start), "s")
    
