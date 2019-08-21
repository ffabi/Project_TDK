import matplotlib.cm as cm
import numpy as np
import os


def to_bgra(depth):
    assert depth.shape[-1] == 1
    depth = depth[:, :, 0]
    jet_img = cm.jet(depth)[..., :3]
    jet_img *= 255
    return jet_img

def equalize_depth_values(depth, cut = 0.4, amin = None, amax = None):
    _depth = depth.copy()
    _depth[_depth < 0] = 0
    if amin is None:
        amin = np.amin(_depth)
        print(amin)
    _depth[_depth < 0.99] -= amin
    if amax is None:
        amax = np.amax(_depth[_depth < cut])
        print(amax)
    if amax != 0:
        _depth[_depth < 0.99] /= amax
    _depth[_depth > 0.99] = 1
    _depth **=.33

    return _depth

