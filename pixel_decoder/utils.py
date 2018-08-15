import os
import sys
from os import path, listdir, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import timeit
# from PIL import Image
import cv2
# from sklearn.model_selection import KFold
# from keras.optimizers import SGD, Adam
# from keras import metrics
# from keras.callbacks import ModelCheckpoint
# from pixel_decoder.resnet_unet import get_resnet_unet
# from pixel_decoder.loss import dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded, dice_logloss
import skimage.io
# import keras.backend as K

def dataformat(fn):
    basename, ext = os.path.splitext(fn)
    return ext

def stats_data(data):
    if len(data.shape) > 3:
        means = np.mean(data, axis = (0, 1, 2))
        stds = np.std(data, axis = (0, 1, 2))
    else:
        means = np.mean(data, axis = (0, 1))
        stds = np.std(data, axis = (0, 1))
    # print(means)
    return means, stds

def color_scale(arr):
    """correct the wv-3 bands to be a composed bands of value between 0 255"""
    axis = (0, 1)
    str_arr = (arr - np.min(arr, axis = axis))*255.0/(np.max(arr, axis = axis) - np.min(arr, axis = axis))
    return str_arr

def open_image(fn):
    format = dataformat(fn)
    if format == '.tif':
        arr = skimage.io.imread(fn, plugin='tifffile').astype('float32')
    else:
        arr = skimage.io.imread(fn).astype('float32')
    img = color_scale(arr)
    return img

def cache_stats(imgs_folder):
    imgs = []
    for f in listdir(path.join(imgs_folder)):
        format = dataformat(f)
        if path.isfile(path.join(imgs_folder, f)) and str(format) in f:
            fpath = path.join(imgs_folder, f)
            img = open_image(fpath)
            img_ = np.expand_dims(img, axis=0)
            imgs.append(img)
    imgs_arr = np.array(imgs)
    dt_means, dt_stds = stats_data(imgs_arr)
    # print("mean for the dataset is {}".format(dt_means))
    # print("Std for the dataset is {}".format(dt_stds))
    return dt_means,dt_stds

def preprocess_inputs_std(x, mean, std):
    """The means and stds are train and validation base.
    It need to be train's stds and means. It might be ok since we are doing KFold split here"""
    zero_msk = (x == 0)
    x = np.asarray(x, dtype='float32')
    x -= mean
    x /= std
    x[zero_msk] = 0
    return x

def datafiles(imgs_folder, masks_folder):
    all_files = []
    all_masks = []
    t0 = timeit.default_timer()
    # imgs_folder = sys.argv[2]
    # masks_folder = os.path.join(os.getcwd(),sys.argv[3])
    # models_folder = os.path.join(os.getcwd(),sys.argv[4])
    for f in sorted(listdir(path.join(os.getcwd(), imgs_folder))):
        if path.isfile(path.join(os.getcwd(),imgs_folder, f)) and dataformat(path.join(os.getcwd(),imgs_folder, f)) in f:
            img_id = f.split('.')[0]
            all_files.append(path.join(os.getcwd(), imgs_folder, f))
            all_masks.append(path.join(masks_folder, '{0}.{1}'.format(img_id, 'png')))
    all_files = np.asarray(all_files)
    all_masks = np.asarray(all_masks)
    return all_files, all_masks

# all_files,all_masks = datafiles(imgs_folder, masks_folder, models_folder)
def rotate_image(image, angle, scale, imgs_folder, masks_folder, models_folder):
    all_files,all_masks = datafiles(imgs_folder, masks_folder)
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result

 # = cache_stats(imgs_folder)

def batch_data_generator(train_idx, batch_size, means, stds, imgs_folder, masks_folder, models_folder, channel_no, border_no, origin_shape_no):
    # all_files, all_masks = datafiles()
    origin_shape = (origin_shape_no, origin_shape_no)
    border = (border_no, border_no)
    all_files, all_masks = datafiles(imgs_folder, masks_folder)
    # means, stds = cache_stats(imgs_folder)
    input_shape = (origin_shape[0] + border[0] + border[1] , origin_shape[1] + border[0] + border[1])
    # input_shape = ()
    inputs = []
    outputs = []
    rgb_index = [0, 1, 2]
    while True:
        np.random.shuffle(train_idx)
        for i in train_idx:
            img = open_image(all_files[i])
            if img.shape[0] != origin_shape[0]:
                img= cv2.resize(img, origin_shape)
            else:
                img = img
            if channel_no == 8:
                img = img
            else:
                band_index = rgb_index
                img = img[:, :, band_index]
            msk = cv2.imread(all_masks[i], cv2.IMREAD_UNCHANGED)[..., 0]

            if random.random() > 0.5:
                scale = 0.9 + random.random() * 0.2
                angle = random.randint(0, 41) - 24
                img = rotate_image(img, angle, scale, imgs_folder, masks_folder, models_folder)
                msk = rotate_image(msk, angle, scale, imgs_folder, masks_folder, models_folder)

            x0 = random.randint(0, img.shape[1] - input_shape[1])
            y0 = random.randint(0, img.shape[0] - input_shape[0])
            img = img[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
            msk = (msk > 127) * 1
            msk = msk[..., np.newaxis]
            otp = msk[y0:y0+input_shape[0], x0:x0+input_shape[1], :]

            if random.random() > 0.5:
                img = img[:, ::-1, ...]
                otp = otp[:, ::-1, ...]

            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                otp = np.rot90(otp, k=rot)

            inputs.append(img)
            outputs.append(otp)

            if len(inputs) == batch_size:
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs, dtype='float')
                inputs = preprocess_inputs_std(inputs, means, stds)
                yield inputs, outputs
                inputs = []
                outputs = []

def val_data_generator(val_idx, batch_size, validation_steps, means, stds, imgs_folder, masks_folder, models_folder, channel_no, border_no, origin_shape_no):
    origin_shape = (origin_shape_no, origin_shape_no)
    border = (border_no, border_no)
    all_files, all_masks = datafiles(imgs_folder, masks_folder)
    # means, stds = cache_stats(imgs_folder)
    input_shape = (origin_shape[0] + border[0] + border[1] , origin_shape[1] + border[0] + border[1])
    all_files,all_masks = datafiles(imgs_folder, masks_folder)
    rgb_index = [0, 1, 2]
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for i in val_idx:
            # img0 = skimage.io.imread(all_files[i], plugin='tifffile')
            img0 = open_image(all_files[i])
            if img0.shape[0] != origin_shape[0]:
                img0= cv2.resize(img0, origin_shape)
            else:
                img0 = img0
            if channel_no == 8:img0 = img0
            else:
                band_index = rgb_index
                img0 = img0[:, :, band_index]
            msk = cv2.imread(all_masks[i], cv2.IMREAD_UNCHANGED)[..., 0:1]
            msk = (msk > 127) * 1
            for x0, y0 in [(0, 0)]:
                img = img0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                otp = msk[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                inputs.append(img)
                outputs.append(otp)
                if len(inputs) == batch_size:
                    step_id += 1
                    inputs = np.asarray(inputs)
                    outputs = np.asarray(outputs, dtype='float')
                    inputs = preprocess_inputs_std(inputs, means, stds)
                    # print(inputs.shape, outputs.shape)
                    # print(np.unique(inputs))
                    yield inputs, outputs
                    inputs = []
                    outputs = []
                    if step_id == validation_steps:
                        break
