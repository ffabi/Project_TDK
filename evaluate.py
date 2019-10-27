import argparse
import tensorflow as tf
from Classes.Evaluator import Evaluator

from Models.OriginalModel import load_trained_model
from keras.callbacks import CSVLogger
import os


def evaluate(args, save_folder):
    model_path = os.path.join(save_folder, "model.h5")
    model = load_trained_model(model_path)

    input_shape = model.layers[0].output.shape
    output_shape = model.layers[-1].output.shape

    evaluator = Evaluator(args, save_folder, input_shape, output_shape)
    evaluator.set_model(model)

    evaluator.export_images(best = True)

    # metrics = evaluator.evaluate_metrics()
    # losses = evaluator.evaluate_losses()
    #
    # metrics.update(losses)
    #
    # on_train_end_csvlogger = CSVLogger("../results/collected_trains_" + str(len(metrics)) + ".csv", separator = ',',
    #                                    append = True)
    # on_train_end_csvlogger.on_train_begin()
    # on_train_end_csvlogger.set_model(model)
    # on_train_end_csvlogger.on_epoch_end(epoch_counter, metrics)
    # on_train_end_csvlogger.on_train_end()


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

    parser = argparse.ArgumentParser(
        description = 'Monocular Depth Estimation on Synthetic Dataset with Dense Ground Truth')

    parser.add_argument('--test_keep', type = float, default = 1, help = '-')

    parser.add_argument('--name', default = '0_decay_08_31_09_20', type = str, help = 'Name prefix')
    parser.add_argument('--shape', type = str, default = '(576, 1024)', help = 'Start training from an existing model.')

    parser.add_argument('--no_shuffle', dest = 'no_shuffle', action = 'store_true', help = 'Turn shuffle off')
    parser.add_argument('--overfit', dest = 'overfit', action = 'store_true', help = 'Overfit on test set')

    parser.add_argument('--input_shape', type = str, default = "(576, 1024, 3)", help = "Start training from an existing model.")
    parser.add_argument('--output_shape', type = str, default = "(576, 1024, 1)", help = "Start training from an existing model.")

    args = parser.parse_args()

    args.shape = eval(args.shape)

    print("args:", vars(args))

    with sess.as_default():

        evaluate(args, "../results/" + args.name)
