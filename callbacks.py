from CustomCallback import *
from keras.callbacks import *


def get_callbacks(args, save_folder):
    
    callbacklist = CallbackList()
    # callbacklist = []
    
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

    # todo later use tensorboard
    # tensorboard = TensorBoard(
    #     log_dir = save_folder,
    #     write_graph = True,
    #     write_grads = False,
    #     write_images = False,
    #     update_freq = 50,
    # )
    # callbacklist.append(tensorboard)

    csvlogger = CSVLogger(os.path.join(save_folder, "train.csv"), separator = ',', append = True)
    callbacklist.append(csvlogger)
    
    return [callbacklist]
