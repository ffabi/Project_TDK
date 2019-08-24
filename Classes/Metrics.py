import numpy as np


# todo use tensors instead
def calculate_eval_metrics(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))

    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()

    avg_rel_err = np.mean(np.abs(gt - pred) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return delta1, delta2, delta3, avg_rel_err, rmse, log_10
