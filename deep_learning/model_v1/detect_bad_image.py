# detect the empty images
# get an error when it reaches out an empty image, the last printed "file_path" is its name.

import os
import sys
import pathlib
from os import makedirs
from os.path import exists, join
import numpy as np
import glob

import tensorflow as tf
# from utils import file_logger
print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


class file_logger():
    def __init__(self, file_path, append=True):
        self.file_path = file_path

        if not append:
            open(self.file_path, 'w').close()

    def __call__(self, string):
        str_aux = string + '\n'

        f = open(self.file_path, 'a')
        f.write(str_aux)
        f.close()

path_save = '/home/xijun/Argonne/save_model/mobilenetv2_binary_nofinetune_nodataaug_checkpoint_best_val_loss_test'
if not exists(path_save):
    makedirs(path_save)

loggers = file_logger(path_save+'/log.txt', False)

loggers('TF version: is {} \n'.format(tf.__version__))
loggers('GPU is {} available\n'.format(str(tf.test.is_gpu_available())))

data_dir = "/home/xijun/Argonne/datasets/dataset_binary"
data_dir = pathlib.Path(data_dir)

loggers('The whole dataset is {}\n'.format(str(data_dir)))

batch_size = 32
img_height = 224
img_width = 224
BATCH_SIZE = batch_size
IMG_SIZE = (img_height, img_width)

image_count = len(list(data_dir.glob('*/*')))
# png for the avian dataset
real_image_count = len(list(data_dir.glob('*/*/*.png')))

loggers('Total tracks number: {}\t Total images number: {}\n'.format(image_count, real_image_count))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

loggers('class_names: {}\n'.format(class_names))

# train: 80%, val: 10%, test: 10%
# ***_ds: separated dataset whose unit is the subfolder (tracks subfolders in avian dataset)
disgard_count = int(image_count * 0.0)
remain_count = image_count - disgard_count

remain_ds = list_ds.skip(disgard_count)

val_size = int(remain_count * 0.2)
train_ds = remain_ds.skip(val_size)
val_ds = remain_ds.take(val_size)
#
test_ds = val_ds.take(int(0.5 * val_size))
val_ds = val_ds.skip(int(0.5 * val_size))

flag_val_ds = 0
flag_train_ds = 0
flag_test_ds = 0

loggers('Remain tracks number: {}\n'.format(remain_count))

loggers('The train_ds (tracks) number: {}\n'.format(tf.data.experimental.cardinality(train_ds).numpy()))
loggers('The val_ds (tracks) number: {}\n'.format(tf.data.experimental.cardinality(val_ds).numpy()))
loggers('The test_ds (tracks) number: {}\n'.format(tf.data.experimental.cardinality(test_ds).numpy()))


# read the images from ***_ds to make the ***_ds_final whose unit is the image, and ***_ds_final is used for model's final training
train_img_list = []
for f in train_ds:
    g = str(f.numpy(), 'utf-8')
    g = pathlib.Path(g)
    train_img_list.extend(glob.glob(str(g / '*.png')))

loggers('The train images number in list : {}\n'.format(len(train_img_list)))
train_ds_final = tf.data.Dataset.list_files(train_img_list, shuffle=False)

train_image_count = tf.data.experimental.cardinality(train_ds_final).numpy()
train_ds_final = train_ds_final.shuffle(train_image_count, reshuffle_each_iteration=False)
loggers('The train images number: {}\n'.format(train_image_count))

val_img_list = []
for f in val_ds:
    g = str(f.numpy(), 'utf-8')
    g = pathlib.Path(g)
    val_img_list.extend(glob.glob(str(g / '*.png')))

loggers('The val images number in list : {}\n'.format(len(val_img_list)))
val_ds_final = tf.data.Dataset.list_files(val_img_list, shuffle=False)

val_image_count = tf.data.experimental.cardinality(val_ds_final).numpy()
val_ds_final = val_ds_final.shuffle(val_image_count, reshuffle_each_iteration=False)
loggers('The val images number: {}\n'.format(val_image_count))

test_img_list = []
for f in test_ds:
    g = str(f.numpy(), 'utf-8')
    g = pathlib.Path(g)
    test_img_list.extend(glob.glob(str(g / '*.png')))

loggers('The test images number in list : {}\n'.format(len(test_img_list)))


test_ds_final = tf.data.Dataset.list_files(test_img_list, shuffle=False)

test_img_list_final = []
for f in test_ds_final:
    g = str(f.numpy(), 'utf-8')
    g = pathlib.Path(g)
    test_img_list_final.extend(glob.glob(str(g)))
loggers('The test images number in the final list : {}\n'.format(len(test_img_list_final)))

test_img_list_diff = list(set(test_img_list) - set(test_img_list_final))


loggers('The difference images number in the test img lists : {}\n'.format(len(test_img_list_diff)))


test_image_count = tf.data.experimental.cardinality(test_ds_final).numpy()
test_ds_final = test_ds_final.shuffle(test_image_count, reshuffle_each_iteration=False)
loggers('The test images number: {}\n'.format(test_image_count))


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The third to last is the class-directory in our avian solar dataset case
    one_hot = tf.dtypes.cast(parts[-3] == class_names, dtype=tf.int16)
    # Integer encode the label
    return tf.argmax(one_hot)

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    print('success decode_jpeg')
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    label = get_label(file_path)
    print('file_path:', file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# AUTOTUNE = tf.data.AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

for ds in train_ds_final:
    process_path(ds)
for ds in val_ds_final:
    process_path(ds)
for ds in test_ds_final:
    process_path(ds)
