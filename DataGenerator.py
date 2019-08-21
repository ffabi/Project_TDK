import keras
import numpy as np
import os, glob
import cv2

import matplotlib.image as mpimg

from helper_functions import *


def depth_converter(depth):
    R = depth[:, :, 0]
    G = depth[:, :, 1]
    B = depth[:, :, 2]
    values = (R + G * 2 ** 8 + B * 2 ** 16) / (2 ** 24 - 1) * 255
    # values = np.array(values, dtype = np.float32)
    return values

    # todo 16 bit?
    # R = depth[:, :, 0]
    # G = depth[:, :, 1]
    # B = depth[:, :, 2]
    # values = (R + G * 2 ** 8 + B * 2 ** 16) / (2 ** 24 - 1)  # * 255 # normalized
    # _depth = np.zeros(depth.shape[:2], np.float32)  # , dtype="uint8"
    # _depth[:, :] = values
    # return _depth.astype(np.float32)


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, set_type, shape, batch_size = 4, shuffle = False, keep_ratio = 1):
        """Initialization"""

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.keep_ratio = keep_ratio
        self.set_type = set_type
        if set_type == "overfit":
            self.set_type = "val"

        # self.resize_name = "synthia"
        self.resize_name = "synthia_resized_" + str(shape[1]) + "x" + str(shape[0])
        self.root = '/root/ffabi_shared_folder/datasets/_structured_datasets/' + self.resize_name
        inputs = glob.glob(os.path.join(self.root, self.set_type, "input_rgb", "SEQ*", "*.png"), recursive = True)
        outputs = glob.glob(os.path.join(self.root, self.set_type, "output_depth", "SEQ*", "*.png"), recursive = True)

        assert len(inputs) == len(outputs), "rgb and depth does not match for the set"
        assert keep_ratio <= 1, "invalid keep ratio"

        self.ids = []
        for p in inputs:
            self.ids.append(p.split("/")[-2] + "/" + p.split("/")[-1])

        self.ids = self.ids[:int(keep_ratio * len(self.ids))]
        self.total_frame_count = round(len(self.ids))

        print(self.set_type, "set has", self.__len__(), "elements")

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
        image_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]

        input_paths = [os.path.join(self.root, self.set_type, "input_rgb", i) for i in image_ids]
        output_paths = [os.path.join(self.root, self.set_type, "output_depth", i) for i in image_ids]

        input_images = []
        for img_path in input_paths:
            img = cv2.imread(img_path)
            input_images.append(img)

        output_images = []
        for img_path in output_paths:
            # todo opencv not reading depth correctly?
            # img = cv2.imread(img_path)
            img = mpimg.imread(img_path)
            img = depth_converter(img)
            output_images.append(img)

        input_images = np.array(input_images)
        output_images = np.array(output_images, dtype = "float32")

        output_images = np.expand_dims(output_images, axis = 3)

        # assert input_images.shape == (self.batch_size, 1080, 1920, 3), "Something went wrong with loading input_image images: {}".format(input_paths)
        # assert output_images.shape == (self.batch_size, 1080, 1920), "Something went wrong with loading output images: {}".format(output_paths)

        # todo debug
        # return np.array(input_images, dtype="float32"), np.array(input_images[:,:,:,:1], dtype="float32")
        return np.array(input_images, dtype = "float32"), output_images

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.ids)


if __name__ == '__main__':
    import tensorflow as tf

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.05)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    with sess.as_default():
        datagen = DataGenerator(set_type = "overfit", batch_size = 4, shuffle = False, shape = (576, 1024))
        print(datagen)
        print("Total frame count:", datagen.total_frame_count)
        batch = datagen.__getitem__(0)
        assert (datagen.__getitem__(0)[0] == datagen.__getitem__(0)[0]).all(), "WTF"

        input_image, output_image = batch[0], batch[1]
        print("Got a sample batch with the input shape of", input_image.shape)
        print("Got a sample batch with the output shape of", output_image.shape)
        print("Max value:", np.amax(batch[1][0]))

        bgra = to_bgra(1 - equalize_depth_values(output_image[0]))

        cv2.imwrite("/root/test.png", bgra)
