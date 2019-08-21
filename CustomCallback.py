import shutil
from collections import OrderedDict

from keras.callbacks import *

from Evaluator import *
from helper_functions import *


class CustomCallback(Callback):
    
    def __init__(self, args, save_folder) -> None:
        super().__init__()
        self.save_folder = save_folder
        self.args = args
        self.evaluator = Evaluator(self.save_folder, args.shape)
        
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
