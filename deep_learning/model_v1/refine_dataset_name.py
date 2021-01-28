# delete the "[" or "]" characters in the image name

import os

data_dir_name = "/home/xijun/Argonne/datasets/dataset_binary"
class_names = os.listdir(data_dir_name)
for class_name in class_names:
    track_names = os.listdir(os.path.join(data_dir_name, class_name))
    for track_name in track_names:
        image_names = os.listdir(os.path.join(data_dir_name, class_name, track_name))
        for image_name in image_names:
            old_file = os.path.join(data_dir_name, class_name, track_name, image_name)
            image_name = image_name.replace('[', '')
            image_name = image_name.replace(']', '')
            new_file = os.path.join(data_dir_name, class_name, track_name, image_name)
            os.rename(old_file, new_file)