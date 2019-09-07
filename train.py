import argparse
from datetime import datetime

from keras.optimizers import Adam

from Classes.Callbacks import *
from Models import OriginalModel
from evaluate import evaluate


def train(args):
    if args.load is not None:
        model_path = os.path.join("../results", args.load, "model.h5")
        model = OriginalModel.load_trained_model(model_path)
        name = args.load + "_continued"

    else:
        print("Creating model with input shape:", args.shape)
        model = OriginalModel.create_model(shape = args.shape)
        # model = simple_dense.create_model(shape = input_shape)

        name = str(args.id) + "_" + args.name + "_" + datetime.now().strftime("%m_%d_%H_%M")

        lossgen = LossGenerator(
            mae_w = 1,
            mse_w = 1,
            first_grad_w = 1,
            second_grad_w = 1,
            ssim_w = 1,
        )

        losses = lossgen.get_losses()
        combined_weighted_loss = losses[-1]

        model.compile(optimizer = Adam(lr = 0.0001, decay = 0), loss = combined_weighted_loss, metrics = losses[:-1])

    train_generator = DataGenerator(
        args = args,
        set_type = "train",
        batch_size = args.batchsize,
        shuffle = not args.no_shuffle,
        keep_ratio = args.train_keep,
        shape = args.shape,
    )
    val_generator = DataGenerator(
        args = args,
        set_type = "val",
        batch_size = 1,
        shuffle = not args.no_shuffle,
        keep_ratio = args.val_keep,
        shape = args.shape,
    )
    assert len(train_generator) and len(val_generator), "No data received from the data generator"

    results_folder = os.path.join("../results", name)

    callbacks = get_callbacks(args, results_folder)

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

    return results_folder


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description = "Monocular Depth Estimation on Synthetic Dataset with Dense Ground Truth")

    parser.add_argument("--id", type = int, default = 0, help = "Train id")
    parser.add_argument("--batchsize", type = int, default = 1, help = "Batch size")
    parser.add_argument("--epochs", type = int, default = 62, help = "Number of epochs")
    parser.add_argument("--eval_freq", type = int, default = 100, help = "todo")

    parser.add_argument("--train_keep", type = float, default = 1, help = "-")
    parser.add_argument("--val_keep", type = float, default = 1, help = "-")
    parser.add_argument("--test_keep", type = float, default = 1, help = "-")

    parser.add_argument("--mae_w", type = float, default = 1, help = "-")
    parser.add_argument("--mse_w", type = float, default = 1, help = "-")
    parser.add_argument("--first_grad_w", type = float, default = 1, help = "-")
    parser.add_argument("--second_grad_w", type = float, default = 1, help = "-")
    parser.add_argument("--ssim_w", type = float, default = 1, help = "-")

    parser.add_argument("--name", default = "debug", type = str, help = "Name prefix")
    parser.add_argument("--load", type = str, help = "Continue training of this train eg: '0_train_08_25_08_34' ")
    parser.add_argument("--shape", type = str, default = "(576, 1024)", help = "Start training from an existing model.")

    parser.add_argument("--no_shuffle", dest = "no_shuffle", action = "store_true", help = "Turn shuffle off")
    parser.add_argument("--overfit", dest = "overfit", action = "store_true", help = "Overfit on test set")

    args = parser.parse_args()

    args.shape = eval(args.shape) + (3,)

    print(vars(args))

    tf.set_random_seed(1234)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

    with sess.as_default():

        results_folder = train(args)
        evaluate(args, results_folder)

    print((time.time() - start), "s")
