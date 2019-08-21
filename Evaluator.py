import os

import cv2
import tensorflow as tf
from keras import Model
from keras.models import model_from_json

from DataGenerator import DataGenerator
from LossGenerator import LossGenerator
from helper_functions import *


class Evaluator:
    model: Model

    def __init__(self, save_folder, shape) -> None:
        super().__init__()
        self.save_folder = save_folder

        self.test_generator = DataGenerator(
            set_type = "test",
            batch_size = 1,
            shuffle = False,
            keep_ratio = 0.1,
            shape = shape
        )

        assert self.test_generator.__len__() != 0, "No data received from the data generator"

        lossgenerator = LossGenerator()
        self.loss_functions = lossgenerator.get_losses()

    def set_model(self, model):
        self.model = model

    def load_model(self, train = "debug"):
        # todo load model from proper json and weight file
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")

        # evaluate loaded model on test data
        loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
        score = loaded_model.evaluate_generator(
            generator = self.test_generator
        )

        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    def evaluate_metrics(self):
        assert self.model is not None, "No model has been set previously"
        self.model.evaluate_generator(self.test_generator,
                                      # callbacks = None, # todo update to
                                      max_queue_size = 10,
                                      workers = 4,
                                      use_multiprocessing = False,
                                      verbose = 1)

    def old_evaluate_metrics(self):
        assert self.model is not None, "No model has been set previously"

        batch = self.test_generator.__getitem__(0)

        rgb_image = batch[0]
        groundtruth = tf.convert_to_tensor(batch[1])
        prediction = tf.convert_to_tensor(self.model.predict(rgb_image))

        print("Test results:")
        for loss_function in self.loss_functions:
            print(loss_function.__name__, loss_function(groundtruth, prediction).eval())

    def export_images(self, epoch):

        batch = self.test_generator.__getitem__(0)
        rgb_images = batch[0]
        groundtruth = batch[1][0]
        prediction = self.model.predict(rgb_images)[0]

        images_path = os.path.join(self.save_folder, "images")
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        epoch__str = "epoch_" + str(epoch)

        print("Saving images")
        cv2.imwrite(os.path.join(images_path, "rgb_input.png"), rgb_images[0])

        bgra_gt = to_bgra(1 - groundtruth)
        cv2.imwrite(os.path.join(images_path, "groundtruth.png"), bgra_gt)

        bgra_gt = to_bgra(1 - equalize_depth_values(groundtruth, 0.4, 0, 0.4))
        cv2.imwrite(os.path.join(images_path, "groundtruth_equalized.png"), bgra_gt)

        bgra_gt = to_bgra(1 - prediction)
        cv2.imwrite(os.path.join(images_path, epoch__str + "prediction.png"), bgra_gt)

        bgra_gt = to_bgra(1 - equalize_depth_values(prediction, 0.4, 0, 0.4))
        cv2.imwrite(os.path.join(images_path, epoch__str + "prediction_equalized.png"), bgra_gt)
        # todo save loss images


if __name__ == '__main__':
    # evaluator = Evaluator(os.path.join("./results", "debug"), shape = (108, 192))
    # # todo
    # # evaluator.load_model()
    # evaluator.evaluate_metrics()
    # evaluator.export_images(0)

    pass
