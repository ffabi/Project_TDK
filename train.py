import argparse
from datetime import datetime

from keras.optimizers import Adam

from Classes.Callbacks import *
from Models import OriginalModel
from evaluate import evaluate


def train(args):
    input_shape = None
    output_shape = None
    if args.input_type == "mono":
        input_shape = (*args.shape, 3)
    elif args.input_type == "stereo":
        input_shape = (*args.shape, 6)

    if args.output_type == "depth":
        output_shape = (*args.shape, 1)
    elif args.output_type == "multitask":
        output_shape = (*args.shape, 2)

    if args.load is not None:
        model_path = os.path.join("../results", args.load, "model.h5")
        model = OriginalModel.load_trained_model(model_path)
        name = args.load + "_continued"

    else:
        print("Creating model with input shape:", input_shape)
        print("Creating model with output shape:", output_shape)
        model = OriginalModel.create_model(input_shape = input_shape, semseg = (args.output_type == "multitask"))
        # model = simple_dense.create_model(shape = input_shape)

        name = str(args.id) + "_" + args.name + "_" + datetime.now().strftime("%m_%d_%H_%M")

        lossgen = LossGenerator(
            mae_w = 1,
            mse_w = 1,
            first_grad_w = 1,
            second_grad_w = 1,
            ssim_w = 1,
            iou_w = 1,
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
        input_type = args.input_type,
        output_type = args.output_type,
    )
    val_generator = DataGenerator(
        args = args,
        set_type = "val",
        batch_size = 1,
        shuffle = not args.no_shuffle,
        keep_ratio = args.val_keep,
        shape = args.shape,
        input_type = args.input_type,
        output_type = args.output_type,
    )
    assert len(train_generator) and len(val_generator), "No data received from the data generator"

    results_folder = os.path.join("../results", name)
    os.makedirs(results_folder)

    args_dict = vars(args)
    print(args_dict)
    with open(os.path.join(results_folder, "args.txt"), 'w') as file:
        file.write(json.dumps(args_dict))

    callbacks = get_callbacks(args, results_folder, input_shape, output_shape)

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

    return results_folder, input_shape, output_shape


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description = "Monocular Depth Estimation on Synthetic Dataset with Dense Ground Truth")

    parser.add_argument("--id", type = int, default = 0, help = "Train id")
    parser.add_argument("--batchsize", type = int, default = 1, help = "Batch size")
    parser.add_argument("--epochs", type = int, default = 100, help = "Number of epochs")

    parser.add_argument("--train_keep", type = float, default = 1, help = "-")
    parser.add_argument("--val_keep", type = float, default = 1, help = "-")
    parser.add_argument("--test_keep", type = float, default = 1, help = "-")

    parser.add_argument("--mae_w", type = float, default = 1, help = "-")
    parser.add_argument("--mse_w", type = float, default = 1, help = "-")
    parser.add_argument("--first_grad_w", type = float, default = 1, help = "-")
    parser.add_argument("--second_grad_w", type = float, default = 1, help = "-")
    parser.add_argument("--ssim_w", type = float, default = 1, help = "-")
    parser.add_argument("--iou_w", type = float, default = 1, help = "-")

    parser.add_argument("--name", default = "debug", type = str, help = "Name prefix")
    parser.add_argument("--load", type = str, help = "Continue training of this train eg: '0_train_08_25_08_34' ")
    parser.add_argument("--shape", type = str, default = "(576, 1024)", help = "Start training from an existing model.")

    parser.add_argument("--input_type", type = str, default = "mono", help = "mono / stereo", choices=["mono", "stereo"])
    parser.add_argument("--output_type", type = str, default = "multitask", help = "depth / multitask", choices = ["depth", "multitask"])

    parser.add_argument("--no_shuffle", dest = "no_shuffle", action = "store_true", help = "Turn shuffle off")
    parser.add_argument("--overfit", dest = "overfit", action = "store_true", help = "Overfit on test set")

    args = parser.parse_args()

    args.shape = eval(args.shape)

    tf.set_random_seed(1234)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

    with sess.as_default():

        results_folder, input_shape, output_shape = train(args)
        evaluate(args, results_folder,)

    print((time.time() - start), "s")
