import multiprocessing
import os, time, cv2

import numpy as np


def single_thread_convert(args_tuple):
    read_path, write_path = args_tuple[0], args_tuple[1]

    # read 32
    depth = cv2.imread(read_path)
    R = depth[:, :, 0]
    G = depth[:, :, 1]
    B = depth[:, :, 2]

    normalized = (R + G * 2 ** 8 + B * 2 ** 16) / (2 ** 24 - 1)  # range: [0; 1]
    upscaled_16 = normalized * (2 ** 16 - 1)  # normalize after read
    upscaled_16 = upscaled_16.astype(np.int32)

    _depth = np.zeros(depth.shape[:2])
    _depth[:, :] = upscaled_16
    _depth = _depth.astype(np.int32)
    cv2.imwrite(write_path, _depth)


def convert_depth(size_x, size_y):
    size_y = int(size_y)
    size_x = int(size_x)
    root_path = '/root/ffabi_shared_folder/datasets/_original_datasets/synthia/'
    read_path = root_path + 'resized_' + str(size_x) + "x" + str(size_y)

    assert os.path.exists(read_path), "Resized dataset does not exist: {}".format(read_path)
    print("Converting resized dataset with shape", size_x, "x", size_y)

    folders = next(os.walk(read_path))[1]
    folders.sort()

    image_paths = []

    folder_types = ["DepthLeft", "DepthRight"]
    for folder in folders:
        for folder_type in folder_types:
            files = os.listdir(os.path.join(read_path, folder, folder_type))
            write_folder = os.path.join(read_path, folder, folder_type + "_converted")

            if not os.path.exists(write_folder):
                os.makedirs(write_folder)

            for file in files:
                read_file = os.path.join(read_path, folder, folder_type, file)
                write_file = os.path.join(write_folder, file)

                image_paths.append((read_file, write_file, size_x, size_y))

    print("Processing")
    pool = multiprocessing.Pool(1)
    pool.map(single_thread_convert, image_paths)
    pool.terminate()
    pool.join()

    print(read_path, "done")


if __name__ == '__main__':
    start = time.time()
    # convert_depth(1920/10, 1080/10)
    # convert_depth(1920 / 1.5, 1080 / 1.5)
    # convert_depth(1920 / 2, 1080 / 2)
    convert_depth(1920 / 4, 1080 / 4)
    # convert_depth(1920 / 8, 1080 / 8)
    print((time.time() - start))
