import argparse
import glob
import os

import cv2
from keras.utils import Sequence
import matplotlib.image as mpimg

from helper_functions import *


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, args, set_type, shape, batch_size, shuffle, keep_ratio, input_type, output_type):
        """Initialization"""

        if args is not None:
            if args.overfit:
                set_type = "test"
                keep_ratio = 1
                print("Overfitting on test set")

        self.input_type = input_type
        self.output_type = output_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.keep_ratio = keep_ratio
        self.set_type = set_type
        self.shape = shape

        self.resize_name = "synthia_resized_" + str(shape[1]) + "x" + str(shape[0])
        self.root = '/root/ffabi_shared_folder/datasets/_structured_datasets/' + self.resize_name

        self.total_frame_count = 0

        self.inputs, self.outputs = self.collect_ids()

    def collect_ids(self):

        inputs = None
        outputs = None

        assert self.input_type in ["mono", "stereo"]
        assert self.output_type in ["depth", "multitask"]

        if self.input_type == "mono" and self.output_type == "depth":

            input_glob = glob.glob(os.path.join(self.root, self.set_type, "RGB*", "SEQ*", "*.png"), recursive = True)
            input_glob = sorted(input_glob)
            inputs = [input_glob]

            output_glob = glob.glob(os.path.join(self.root, self.set_type, "Depth*", "SEQ*", "*.png"), recursive = True)
            output_glob = sorted(output_glob)
            outputs = [output_glob]

        elif self.input_type == "stereo" and self.output_type == "depth":
            rgb_left = glob.glob(os.path.join(self.root, self.set_type, "RGBLeft", "SEQ*", "*.png"), recursive = True)
            rgb_left = sorted(rgb_left)

            rgb_right = glob.glob(os.path.join(self.root, self.set_type, "RGBRight", "SEQ*", "*.png"), recursive = True)
            rgb_right = sorted(rgb_right)

            inputs = [rgb_left, rgb_right]

            depth_left = glob.glob(os.path.join(self.root, self.set_type, "DepthLeft", "SEQ*", "*.png"),
                                   recursive = True)
            depth_left = sorted(depth_left)
            outputs = [depth_left]

        elif self.input_type == "mono" and self.output_type == "multitask":
            input_glob = glob.glob(os.path.join(self.root, self.set_type, "RGB*", "SEQ*", "*.png"), recursive = True)
            input_glob = sorted(input_glob)
            inputs = [input_glob]

            depth = glob.glob(os.path.join(self.root, self.set_type, "Depth*", "SEQ*", "*.png"), recursive = True)
            depth = sorted(depth)
            semseg = glob.glob(os.path.join(self.root, self.set_type, "GT*", "SEQ*", "*.png"), recursive = True)
            semseg = sorted(semseg)
            outputs = [depth, semseg]

        elif self.input_type == "stereo" and self.output_type == "multitask":
            rgb_left = glob.glob(os.path.join(self.root, self.set_type, "RGBLeft", "SEQ*", "*.png"), recursive = True)
            rgb_left = sorted(rgb_left)

            rgb_right = glob.glob(os.path.join(self.root, self.set_type, "RGBRight", "SEQ*", "*.png"), recursive = True)
            rgb_right = sorted(rgb_right)

            inputs = [rgb_left, rgb_right]

            depth = glob.glob(os.path.join(self.root, self.set_type, "DepthLeft", "SEQ*", "*.png"), recursive = True)
            depth = sorted(depth)
            semseg = glob.glob(os.path.join(self.root, self.set_type, "GTLeft", "SEQ*", "*.png"), recursive = True)
            semseg = sorted(semseg)
            outputs = [depth, semseg]

        assert inputs and outputs

        length = len(inputs[0])

        for input, output in zip(inputs, outputs):
            assert len(input) == length and len(output) == length, "dataset sizes do not match"
        assert self.keep_ratio <= 1, "invalid keep ratio"

        adjusted_length = round(self.keep_ratio * length)
        self.total_frame_count = adjusted_length

        inputs = inputs[:int(adjusted_length)]
        outputs = outputs[:int(adjusted_length)]

        print(self.set_type, "set has", adjusted_length, "elements")
        return np.array(inputs), np.array(outputs)

        # self.ids = []
        # for p in inputs:
        #     self.ids.append(p.split("/")[-2] + "/" + p.split("/")[-1])
        # print(self.ids)
        # self.ids = self.ids[:int(self.keep_ratio * len(self.ids))]
        # self.total_frame_count = round(length)

    def __str__(self):
        exclude = ["ids"]
        return str(type(self)) + " " + ", ".join(
            "%s: %s" % item for item in vars(self).items() if str(item[0]) not in exclude)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.total_frame_count / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        assert index < self.__len__(), "index is out of range"
        inputs = self.inputs[..., index * self.batch_size:(index + 1) * self.batch_size]
        outputs = self.outputs[..., index * self.batch_size:(index + 1) * self.batch_size]

        input_images = []
        for input in inputs.transpose():
            batch_images = []
            for image_path in input:
                img = cv2.imread(image_path)
                batch_images.append(img)

            batch_images = np.concatenate(batch_images, axis = 2)
            input_images.append(batch_images)

        output_images = []

        if self.output_type == "multitask":
            depths = outputs[0]
            semsegs = outputs[1]

            for depth_semseg_pair in zip(depths, semsegs):
                batch_images = []

                img = mpimg.imread(depth_semseg_pair[0])
                img = depth_converter(img)
                batch_images.append(img)

                img = mpimg.imread(depth_semseg_pair[1])
                img = semseg_to_onehot_converter(img)
                batch_images.append(img)

                batch_images = np.concatenate(batch_images, axis = 2)
                output_images.append(batch_images)
        else:
            depths = outputs[0]
            for depth_semseg_pair in depths:
                batch_images = []

                img = mpimg.imread(depth_semseg_pair)
                img = depth_converter(img)
                batch_images.append(img)

                batch_images = np.concatenate(batch_images, axis = 2)
                output_images.append(batch_images)

        input_images = np.array(input_images, dtype = "float32")
        output_images = np.array(output_images, dtype = "float32")
        # output_images = output_images.reshape((self.batch_size, *self.shape, -1))

        return input_images, output_images

    def on_epoch_end(self):
        if self.shuffle:
            # todo shuffle input and output together
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Monocular Depth Estimation on Synthetic Dataset with Dense Ground Truth")

    parser.add_argument("--name", default = "debug", type = str, help = "Name prefix")
    parser.add_argument("--load", type = str, help = "Continue training of this train eg: '0_train_08_25_08_34' ")
    parser.add_argument("--shape", type = str, default = "(576, 1024)", help = "Start training from an existing model.")

    parser.add_argument("--overfit", dest = "overfit", action = "store_true", help = "Overfit on test set")

    args = parser.parse_args()

    args.shape = eval(args.shape)

    import tensorflow as tf

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.05)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    with sess.as_default():
        datagen = DataGenerator(args = args, set_type = "train", batch_size = 4, shuffle = False, shape = (576, 1024),
                                input_type = "mono", output_type = "multitask", keep_ratio = 1)

        print("Dir", datagen.root)
        print("Total frame count:", datagen.total_frame_count)
        batch = datagen.__getitem__(0)
        # assert (datagen.__getitem__(0)[0] == datagen.__getitem__(0)[0]).all(), "WTF"

        input, output = batch[0], batch[1]
        print("Got a sample batch with the input shape of", input.shape)
        print("Got a sample batch with the output shape of", output.shape)

        depth_bgra = to_bgra(1 - equalize_depth_values(output[0][..., :1]))

        cv2.imwrite("/root/test_rgb.png", input[0][..., :3])
        cv2.imwrite("/root/test_depth.png", depth_bgra)

        ped_mask = output[0][..., 1:][..., 12] == 1
        car_mask = output[0][..., 1:][..., 14] == 1
        base = output[0][..., 1:][..., 0] != 42

        weight_mask = tf.cast(ped_mask, tf.float32) * 9 + tf.cast(car_mask, tf.float32) * 9 + tf.cast(base, tf.float32)
        # semseg = onehot_to_semseg_converter(output[0][..., 1:])
        # cv2.imwrite("/root/test_semseg.png", semseg)
