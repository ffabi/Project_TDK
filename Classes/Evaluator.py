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

    def export_images(self, epoch):

        batch = self.test_generator.__getitem__(0)
        rgb_images = batch[0]
        groundtruth = batch[1][0]
        predictions = self.model.predict(rgb_images)
        prediction = predictions[0]

        image_types = ["prediction", "prediction_equalized", "absolute_error", "first_derivative_error",
                       "second_derivative_error"]
        images_path = os.path.join(self.save_folder, "images")
        for images_type in image_types:
            if not os.path.exists(os.path.join(images_path, images_type)):
                os.makedirs(os.path.join(images_path, images_type))

        epoch_str = "epoch_" + str(epoch) + "_"

        print("Saving images")
        if epoch == 1:
            cv2.imwrite(os.path.join(images_path, "rgb_input.png"), rgb_images[0])

            bgra_gt = to_bgra(1 - groundtruth)
            cv2.imwrite(os.path.join(images_path, "groundtruth.png"), bgra_gt)

            bgra_gt = to_bgra(1 - equalize_depth_values(groundtruth, 0.4, 0, 0.4))
            cv2.imwrite(os.path.join(images_path, "groundtruth_equalized.png"), bgra_gt)

        # todo implement multiprocessing write_image(image, image_path, name = "pred", folder = "")

        bgra_gt = to_bgra(1 - prediction)
        cv2.imwrite(os.path.join(images_path, "prediction", epoch_str + "prediction.png"), bgra_gt)

        bgra_gt = to_bgra(1 - equalize_depth_values(prediction, 0.4, 0, 0.4))
        cv2.imwrite(os.path.join(images_path, "prediction_equalized", epoch_str + "prediction_equalized.png"), bgra_gt)

        groundtruth = tf.convert_to_tensor(batch[1])
        prediction = tf.convert_to_tensor(predictions)

        absolute_error_image = get_absolute_error_image(groundtruth, prediction).eval()[0]
        bgra_gt = to_bgra(1 - absolute_error_image)
        cv2.imwrite(os.path.join(images_path, "absolute_error", epoch_str + "absolute_error.png"), bgra_gt)

        first_derivative_error_image = get_first_derivative_error_image(groundtruth, prediction).eval()[0]
        bgra_gt = to_bgra(1 - first_derivative_error_image)
        cv2.imwrite(os.path.join(images_path, "first_derivative_error", epoch_str + "first_derivative_error.png"),
                    bgra_gt)

        second_derivative_error_image = get_second_derivative_error_image(groundtruth, prediction).eval()[0]
        bgra_gt = to_bgra(1 - second_derivative_error_image)
        cv2.imwrite(os.path.join(images_path, "second_derivative_error", epoch_str + "second_derivative_error.png"),
                    bgra_gt)


if __name__ == '__main__':
    # evaluator = Evaluator(os.path.join("./results", "debug"), shape = (108, 192))
    # # todo
    # # evaluator.load_model()
    # evaluator.evaluate_metrics()
    # evaluator.export_images(0)

    pass
