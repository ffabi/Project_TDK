import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import cv2, os
import glob, collections

dataset_path = '/root/ffabi_shared_folder/datasets/_original_datasets/synthia/SYNTHIA-SF/'
sample = "0000000"

def depth_converter(depth):
    R = depth[:, :, 0]
    G = depth[:, :, 1]
    B = depth[:, :, 2]
    values = (R + G * 2 ** 8 + B * 2 ** 16) / (2 ** 24 - 1)
    # values = np.array(values, dtype = np.float32)
    return values


def equalize_depth_values(depth_image_1d, cut = 0.4, amin = None, amax = None):
    depth = depth_image_1d
    depth[depth < 0.99] -= np.amin(depth_image_1d)
    if amax is None:
        amax = np.amax(depth[depth < cut])
    depth[depth < 0.99] /= amax
    depth[depth > 0.99] = 1
    depth **=.33

    return depth

def to_bgra(depth, invalid_value = 0.999):
    jet_img = cm.jet(depth)[..., :3]
    jet_img *= 255
    return jet_img

# depth = mpimg.imread(dataset_path + 'SEQ1/DepthLeft/'+sample+'.png')
depth = cv2.imread(dataset_path + 'SEQ1/DepthLeft/'+sample+'.png')

bgra = to_bgra(equalize_depth_values(depth_converter(depth)))

cv2.imwrite("/root/test.png", bgra)