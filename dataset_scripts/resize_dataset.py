import multiprocessing
import os, time, cv2


def single_thread_resize(args_tuple):
    read_path, write_path = args_tuple[0], args_tuple[1]
    size_x, size_y = args_tuple[2], args_tuple[3]

    image_file = cv2.imread(read_path)
    new_image = cv2.resize(image_file, (size_x, size_y))
    cv2.imwrite(write_path, new_image)

def create_resized_dataset(size_x, size_y, forced = False):
    size_y = int(size_y)
    size_x = int(size_x)
    root_path = '/root/ffabi_shared_folder/datasets/_original_datasets/synthia/'
    read_path = root_path + 'SYNTHIA-SF/'
    write_path = root_path + 'resized_' + str(size_x) + "x" + str(size_y)
    
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    elif not forced:
        # todo check number of files in the old set
        print("Resized dataset already exists, use parameter 'forced = True' to overwrite it")
        return
    print("Creating resized dataset with shape", size_x, "x", size_y)
    
    folders = next(os.walk(read_path))[1]
    folders.sort()
    
    image_paths = []
    
    folder_types = ["RGBLeft", "RGBRight", "DepthLeft", "DepthRight"]
    for folder in folders:
        for folder_type in folder_types:
            files = os.listdir(os.path.join(read_path, folder, folder_type))
            write_folder = os.path.join(write_path, folder, folder_type)
            
            if not os.path.exists(write_folder):
                os.makedirs(write_folder)
                
            for file in files:
                read_file = os.path.join(read_path, folder, folder_type, file)
                write_file = os.path.join(write_folder, file)
                
                image_paths.append((read_file, write_file, size_x, size_y))
        
    
    pool = multiprocessing.Pool(16)
    pool.map(single_thread_resize, image_paths)
    pool.terminate()
    pool.join()
    
    print(write_path, "done")


if __name__ == '__main__':
    start = time.time()
    # resize(1920/10, 1080/10)
    # create_resized_dataset(1920 / 1.5, 1080 / 1.5)
    # create_resized_dataset(1920 / 2, 1080 / 2)
    # create_resized_dataset(1920 / 4, 1080 / 4)
    # create_resized_dataset(1920 / 8, 1080 / 8)
    create_resized_dataset(1024, 576)
    # create_resized_dataset(1920 / 8, 1080 / 8)
    print((time.time() - start))
