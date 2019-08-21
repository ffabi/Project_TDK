import shutil

from keras.callbacks import *

from Evaluator import Evaluator


# class CustomTensorBoard(TensorBoard):
#
#     def __init__(self, log_dir):  # add other arguments to __init__ if you need
#         super().__init__(log_dir = log_dir,
#                          write_graph = False,
#                          write_grads = False,
#                          write_images = False,
#                          )
#
#     def on_epoch_end(self, epoch, logs = None):
#         logs.update({"lr": K.eval(self.model.optimizer.lr)})
#         super().on_epoch_end(epoch, logs)


class CallbackGenerator:

    def __init__(self, folder_prefix, shape) -> None:
        super().__init__()
        self.shape = shape
        self.save_folder = os.path.join("../results", folder_prefix)

        self.evaluator = Evaluator(self.save_folder, self.shape)

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.model = None

        self.earlystop = EarlyStopping(
            monitor = "val_loss",
            min_delta = 0.0001,
            patience = 10,
            verbose = 1,
            mode = "auto",
            restore_best_weights = True,
        )

        # we are going to keep only the best model
        self.mcp = ModelCheckpoint(
            filepath = os.path.join(self.save_folder, "weights.h5"),
            verbose = 1,
            save_best_only = True,
        )

        # to be sure that we stop if the gradient explodes, etc
        self.ton = TerminateOnNaN()

        self.tensorboard = TensorBoard(
            log_dir = self.save_folder,
            write_graph = True,
            write_grads = False,
            write_images = False,
            update_freq = 50,
        )

        self.lambdacallback = LambdaCallback(
            on_train_begin = self.on_train_begin,
            on_epoch_end = self.on_epoch_end,
            on_train_end = self.on_train_end,
        )

    # todo customcallback

    def on_train_begin(self, logs):
        assert self.model is not None, "No model has been set previously"

        model_json = self.model.to_json()
        with open(os.path.join(self.save_folder, "model.json"), "w") as json_file:
            json_file.write(model_json)

        shutil.copytree(".", os.path.join(self.save_folder, "source"))


    def on_epoch_end(self, epoch, logs = None):
        self.evaluator.export_images(epoch)

    def on_train_end(self, logs):
        self.evaluator.evaluate_metrics()

    def set_model(self, model):
        self.model = model
        self.evaluator.set_model(model = self.model)


    def get_callbacks(self):
        return [self.earlystop, self.mcp, self.ton, self.lambdacallback]
        # return [self.earlystop, self.mcp, self.ton, self.tensorboard, self.lambdacallback]
