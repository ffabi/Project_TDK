# TODO split resized sets
import matplotlib.image as mpimg
import os
import subprocess
import glob

size = "resized_1024x576/"

synthia_sequences = '/root/ffabi_shared_folder/datasets/_original_datasets/synthia/' + size
structured = '/root/ffabi_shared_folder/datasets/_structured_datasets/synthia_' + size
sample = "0000000"

train = ['SEQ2', 'SEQ5', 'SEQ6', 'SEQ4']
test = ['SEQ3']
val = ['SEQ1']

sets = [(train, "train"), (test, "test"), (val, "val")]

for s in sets:
    if not os.path.exists(os.path.join(structured, s[1])):
        os.makedirs(os.path.join(structured, s[1]))
    
    if not os.path.exists(os.path.join(structured, s[1], "input_rgb")):
        os.mkdir(os.path.join(structured, s[1], "input_rgb"))
    
    if not os.path.exists(os.path.join(structured, s[1], "output_depth")):
        os.mkdir(os.path.join(structured, s[1], "output_depth"))

for set in sets:
    for seq in set[0]:
        dst = os.path.join(synthia_sequences, seq, "RGBLeft")
        src = os.path.join(structured, set[1], "input_rgb", seq + "_Left")
        if os.path.exists(src):
            os.remove(src)
        os.symlink(dst, src)
        
        dst = os.path.join(synthia_sequences, seq, "RGBRight")
        src = os.path.join(structured, set[1], "input_rgb", seq + "_Right")
        if os.path.exists(src):
            os.remove(src)
        os.symlink(dst, src)
        
        dst = os.path.join(synthia_sequences, seq, "DepthLeft")
        src = os.path.join(structured, set[1], "output_depth", seq + "_Left")
        if os.path.exists(src):
            os.remove(src)
        os.symlink(dst, src)
        
        dst = os.path.join(synthia_sequences, seq, "DepthRight")
        src = os.path.join(structured, set[1], "output_depth", seq + "_Right")
        if os.path.exists(src):
            os.remove(src)
        os.symlink(dst, src)




