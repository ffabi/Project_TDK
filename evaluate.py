import argparse
import tensorflow as tf
from Classes.Evaluator import Evaluator

from Models.OriginalModel import load_trained_model
from keras.callbacks import CSVLogger
import os


def evaluate(args, save_folder):
    with sess.as_default():
        model_path = os.path.join(save_folder, "model.h5")
        model = load_trained_model(model_path)

        evaluator = Evaluator(args, save_folder)
        evaluator.set_model(model)

        # evaluator.evaluate_metrics()
        # evaluator.evaluate_losses()

        evaluator.export_images(best = True)

        # evaluator.export_images("best")



        # all_values = OrderedDict(vars(args))
        # all_values.update(metrics)
        #
        # on_train_end_csvlogger = CSVLogger("../results/collected_trains_" + str(len(all_values)) + ".csv", separator = ',',
        #                                    append = True)
        # on_train_end_csvlogger.on_train_begin()
        # on_train_end_csvlogger.set_model(model)
        # on_train_end_csvlogger.on_epoch_end(epoch_counter, all_values)
        # on_train_end_csvlogger.on_train_end()


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

    parser = argparse.ArgumentParser(
        description = 'Monocular Depth Estimation on Synthetic Dataset with Dense Ground Truth')

    parser.add_argument('--test_keep', type = float, default = 1, help = '-')

    parser.add_argument('--name', default = 'second_long_3_0_08_22_20_54', type = str, help = 'Name prefix')
    parser.add_argument('--shape', type = str, default = '(576, 1024)', help = 'Start training from an existing model.')

    parser.add_argument('--no_shuffle', dest = 'no_shuffle', action = 'store_true', help = 'Turn shuffle off')
    parser.add_argument('--overfit', dest = 'overfit', action = 'store_true', help = 'Overfit on test set')

    args = parser.parse_args()

    args.shape = eval(args.shape)

    print("args:", vars(args))
    evaluate(args, "../results/" + args.name)
