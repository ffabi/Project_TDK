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
        
        self.model = None
        self.epoch_counter = 0
    
    
    def on_train_begin(self, logs = None):
        
        model_json = self.model.to_json()
        with open(os.path.join(self.save_folder, "model.json"), "w") as json_file:
            json_file.write(model_json)
        
        dst = os.path.join(self.save_folder, "source")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(".", dst)
    
    
    def on_epoch_end(self, epoch, logs = None):
        
        # to be used in on_train_end
        self.epoch_counter = epoch + 1
        
        self.evaluator.export_images(epoch + 1)
    
    
    def on_train_end(self, logs = None):
        metrics = self.evaluator.evaluate_metrics()
        
        all_values = OrderedDict(vars(self.args))
        all_values.update(metrics)
        
        on_train_end_csvlogger = CSVLogger("../results/collected_trains_" + str(len(all_values)) + ".csv", separator = ',', append = True)
        on_train_end_csvlogger.on_train_begin()
        on_train_end_csvlogger.set_model(self.model)
        on_train_end_csvlogger.on_epoch_end(self.epoch_counter, all_values)
        on_train_end_csvlogger.on_train_end()
    
    
    def set_model(self, model):
        self.model = model
        self.evaluator.set_model(model = model)


def get_callbacks(args, save_folder):
    callbacklist = CallbackList()

    customcallback = CustomCallback(args, save_folder)
    callbacklist.append(customcallback)

    earlystop = EarlyStopping(
        monitor = "val_loss",
        min_delta = 0.0001,
        patience = 10,
        verbose = 1,
        mode = "auto",
        restore_best_weights = True,
    )
    callbacklist.append(earlystop)

    # we are going to keep only the best model
    mcp = ModelCheckpoint(
        filepath = os.path.join(save_folder, "weights.h5"),
        verbose = 1,
        save_best_only = True,
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