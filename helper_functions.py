import matplotlib.cm as cm
import numpy as np

import tensorflow as tf

def to_bgra(depth):
    assert depth.shape[-1] == 1
    depth = depth[:, :, 0]
    jet_img = cm.jet(depth)[..., :3]
    jet_img *= 255
    return jet_img

def equalize_depth_values(depth, cut = 0.4, amin = None, amax = None):
    _depth = depth.copy()
    _depth[_depth < 0] = 0
    amin = amin or np.amin(_depth)
    _depth[_depth < 0.99] -= amin
    amax = amax or np.amax(_depth[_depth < cut])
    if amax != 0:
        _depth[_depth < 0.99] /= amax
    _depth[_depth > 0.99] = 1
    _depth **=.33

    return _depth


def depth_converter(depth):
    R = depth[:, :, 0]
    G = depth[:, :, 1]
    B = depth[:, :, 2]
    values = (R + G * 2 ** 8 + B * 2 ** 16) / (2 ** 24 - 1) * 255
    # values = np.array(values, dtype = np.float32)
    return np.expand_dims(values, axis = 2)

    # todo 16 bit?
    # R = depth[:, :, 0]
    # G = depth[:, :, 1]
    # B = depth[:, :, 2]
    # values = (R + G * 2 ** 8 + B * 2 ** 16) / (2 ** 24 - 1)  # * 255 # normalized
    # _depth = np.zeros(depth.shape[:2], np.float32)  # , dtype="uint8"
    # _depth[:, :] = values
    # return _depth.astype(np.float32)


def semseg_to_onehot_converter(semseg):
    seg_labels = np.zeros((semseg.shape[0], semseg.shape[1], 23))

    semseg = semseg[:, :, 0]
    semseg = np.round(semseg * 255)
    # semseg = tf.cast(semseg, dtype = int)

    for c in range(23):
        seg_labels[:, :, c] = (semseg == c).astype(int)

    return seg_labels


def onehot_to_semseg_converter(onehot_semseg):
    semseg = np.zeros(shape = (*onehot_semseg.shape[:-1], 3))

    onehot_semseg = np.round(onehot_semseg)

    for i in range(len(semseg)):
        for j in range(len(semseg[i])):
            for c in range(23):
                if onehot_semseg[i, j, c] == 1:
                    semseg[i, j, 2] = c

    return semseg


    # base_mask = onehot_semseg[..., 0] == 1
    # base_mask = tf.cast(base_mask, tf.float32)
    #
    # for i in range(1,23):
    #     mask = onehot_semseg[..., i] == 1
    #     base_mask += tf.cast(mask, tf.float32)
    #
    # return base_mask.eval()


