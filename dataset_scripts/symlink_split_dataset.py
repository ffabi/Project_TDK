import os

size = "resized_1024x576/"

synthia_sequences = '/root/ffabi_shared_folder/datasets/_original_datasets/synthia/' + size
structured = '/root/ffabi_shared_folder/datasets/_structured_datasets/synthia_' + size

train = ['SEQ2', 'SEQ5', 'SEQ6', 'SEQ4']
test = ['SEQ3']
val = ['SEQ1']

sets = [(train, "train"), (test, "test"), (val, "val")]
folder_types = ["RGBLeft", "RGBRight", "DepthLeft", "DepthRight", "GTLeft", "GTRight"]
for s in sets:
    if not os.path.exists(os.path.join(structured, s[1])):
        os.makedirs(os.path.join(structured, s[1]))
    for folder_type in folder_types:
        if not os.path.exists(os.path.join(structured, s[1], folder_type)):
            os.mkdir(os.path.join(structured, s[1], folder_type))


for set in sets:
    for seq in set[0]:
        for folder_type in folder_types:

            original_path = os.path.join(synthia_sequences, seq, folder_type)
            structured_path = os.path.join(structured, set[1], folder_type, seq + "_" + folder_type)

            print(structured_path, original_path)


            os.symlink(original_path, structured_path)
