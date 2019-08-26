import shutil

from keras.callbacks import *

from Classes.Evaluator import *
from helper_functions import *


class CustomCallback(Callback):

    def __init__(self, args, save_folder) -> None:
        super().__init__()
        self.save_folder = save_folder
        self.args = args
        self.evaluator = Evaluator(args, self.save_folder)

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def set_model(self, model):
        self.evaluator.set_model(model = model)

    def on_train_begin(self, logs = None):

        dst = os.path.join(self.save_folder, "source")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(".", dst)



    def on_epoch_end(self, epoch, logs = None):
        self.evaluator.export_images(epoch + 1)


def get_callbacks(args, save_folder):
    callbacklist = CallbackList()

    customcallback = CustomCallback(args, save_folder)
    callbacklist.append(customcallback)

    earlystop = EarlyStopping(
        monitor = "val_loss",
        min_delta = 0.0001,
        patience = 16,
        verbose = 1,
        mode = "auto",
        restore_best_weights = True,
    )
    callbacklist.append(earlystop)

    # we are going to keep only the best model
    mcp = ModelCheckpoint(
        filepath = os.path.join(save_folder, "model.h5"),
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
    )

    callbacklist.append(mcp)

    # to be sure that we stop if the gradient explodes, etc
    ton = TerminateOnNaN()
    callbacklist.append(ton)

    tensorboard = TensorBoard(
        log_dir = save_folder,
        write_graph = True,
        write_grads = True,
        write_images = False,
        update_freq = 1000,
    )
    callbacklist.append(tensorboard)

    csvlogger = CSVLogger(os.path.join(save_folder, "train.csv"), separator = ',', append = True)
    callbacklist.append(csvlogger)

    return [callbacklist]
