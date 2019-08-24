import cv2
from keras import Model
from keras.models import model_from_json

from Classes.DataGenerator import DataGenerator
from Classes.LossGenerator import *
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
        metrics = self.model.evaluate_generator(self.test_generator,
                                                # callbacks = None, # todo update libraries
                                                max_queue_size = 10,
                                                workers = 4,
                                                use_multiprocessing = False,
                                                verbose = 1)

        return dict(zip(self.model.metrics_names, metrics))
    
    def old_evaluate_metrics(self):
        assert self.model is not None, "No model has been set previously"
        
        batch = self.test_generator.__getitem__(0)
        
        rgb_images = batch[0]
        groundtruth = tf.convert_to_tensor(batch[1])
        prediction = tf.convert_to_tensor(self.model.predict(rgb_images))
        
        print("Test results:")
        for loss_function in self.loss_functions:
            print(loss_function.__name__, loss_function(groundtruth, prediction).eval())
    
    def export_images(self, epoch):
    
        batch = self.test_generator.__getitem__(0)
        rgb_images = batch[0]
        groundtruth = batch[1][0]
        predictions = self.model.predict(rgb_images)
        prediction = predictions[0]
    
        image_types = ["prediction", "prediction_equalized", "absolute_error", "first_derivative_error", "second_derivative_error"]
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
        cv2.imwrite(os.path.join(images_path, "first_derivative_error", epoch_str + "first_derivative_error.png"), bgra_gt)

        second_derivative_error_image = get_second_derivative_error_image(groundtruth, prediction).eval()[0]
        bgra_gt = to_bgra(1 - second_derivative_error_image)
        cv2.imwrite(os.path.join(images_path, "second_derivative_error", epoch_str + "second_derivative_error.png"), bgra_gt)

if __name__ == '__main__':
    # evaluator = Evaluator(os.path.join("./results", "debug"), shape = (108, 192))
    # # todo
    # # evaluator.load_model()
    # evaluator.evaluate_metrics()
    # evaluator.export_images(0)
    
    pass
