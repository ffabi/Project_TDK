import cv2
from keras import Model

from Classes.DataGenerator import DataGenerator
from Classes.LossGenerator import *
from Classes.Metrics import get_eval_metrics
from helper_functions import *


class Evaluator:
    model: Model

    def __init__(self, args, save_folder) -> None:
        super().__init__()
        self.save_folder = save_folder

        self.test_generator = DataGenerator(
            args = args,
            set_type = "test",
            batch_size = 1,
            shuffle = False,
            keep_ratio = args.test_keep,
            shape = args.shape
        )

        assert len(self.test_generator), "No data received from the data generator"

        lossgenerator = LossGenerator()
        self.loss_functions = lossgenerator.get_losses()
        self.metric_functions = get_eval_metrics()

    def set_model(self, model):
        self.model = model

    def evaluate_metrics(self):

        assert self.model is not None, "No model has been set previously"
        print("Evaluating metrics on test set")

        result_names = []
        result_values = []

        for i in range(len(self.test_generator)):
            batch = self.test_generator.__getitem__(i)
            groundtruth = batch[1][0]
            prediction = self.model.predict(batch[0])[0]

            batch_result_values = []
            for metric_function in self.metric_functions:
                if i == 0 and metric_function.__name__ not in result_names:
                    result_names.append(metric_function.__name__)
                batch_result_values.append(metric_function(groundtruth, prediction))
            result_values.append(batch_result_values)

        result_values = np.mean(result_values, axis = 0)
        results = dict(zip(result_names, result_values))

        print("Metrics:", results)

        return results

    def evaluate_losses(self):
        assert self.model is not None, "No model has been set previously"
        print("Evaluating losses on test set")

        result_names = []
        result_values = []

        for i in range(len(self.test_generator)):
            batch = self.test_generator.__getitem__(i)
            groundtruth = batch[1][0]
            prediction = self.model.predict(batch[0])[0]

            batch_result_values = []
            for loss_function in self.loss_functions:
                if i == 0 and loss_function.__name__ not in result_names:
                    result_names.append(loss_function.__name__)
                batch_result_values.append(loss_function(groundtruth, prediction).eval())
            result_values.append(batch_result_values)

        result_values = np.mean(result_values, axis = 0)
        results = dict(zip(result_names, result_values))

        print("Losses:", results)

        return results

    def save_image(self, image, images_path, best, epoch_str, name, equalize):
        # todo Adaptive Histogram Equalization: https://towardsdatascience.com/histogram-equalization-5d1013626e64
        if equalize:
            image = equalize_depth_values(image, 0.4, 0, 0.4)

        image = 1 - image

        bgra = to_bgra(image)

        if best or epoch_str == "":
            folder_name = ""
        else:
            folder_name = name

        folder = os.path.join(images_path, folder_name)
        os.path.exists(folder) or os.makedirs(folder)
        cv2.imwrite(os.path.join(folder, epoch_str + name + ".png"), bgra)

    def export_images(self, epoch = 0, best = False):
        print("Saving images")

        batch = self.test_generator.__getitem__(0)
        rgb_images = batch[0]
        groundtruth = batch[1][0]
        predictions = self.model.predict(rgb_images)
        prediction = predictions[0]

        if best:
            epoch_str = ""
            save_folder = os.path.join(self.save_folder, "best")

        else:
            epoch_str = "epoch_" + str(epoch) + "_"
            save_folder = os.path.join(self.save_folder, "images")

        self.save_image(groundtruth, save_folder, best, "", "groundtruth", False)
        self.save_image(groundtruth, save_folder, best, "", "groundtruth_equalized", True)

        self.save_image(prediction, save_folder, best, "", "prediction", False)
        self.save_image(prediction, save_folder, best, "", "prediction_equalized", True)

        cv2.imwrite(os.path.join(save_folder, "rgb_input.png"), rgb_images[0])

        groundtruth = tf.convert_to_tensor(batch[1])
        prediction = tf.convert_to_tensor(predictions)

        absolute_error_image = get_absolute_error_image(groundtruth, prediction).eval()[0]
        first_derivative_error_image = get_first_derivative_error_image(groundtruth, prediction).eval()[0]
        second_derivative_error_image = get_second_derivative_error_image(groundtruth, prediction).eval()[0]

        self.save_image(absolute_error_image, save_folder, best, epoch_str, "absolute_error_image", False)
        self.save_image(absolute_error_image, save_folder, best, epoch_str, "absolute_error_image_equalized", True)

        self.save_image(first_derivative_error_image, save_folder, best, epoch_str, "first_derivative_error_image",
                        False)
        self.save_image(first_derivative_error_image, save_folder, best, epoch_str,
                        "first_derivative_error_image_equalized", True)

        self.save_image(second_derivative_error_image, save_folder, best, epoch_str, "second_derivative_error_image",
                        False)
        self.save_image(second_derivative_error_image, save_folder, best, epoch_str,
                        "second_derivative_error_image_equalized", True)
