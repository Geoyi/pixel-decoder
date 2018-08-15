# -*- coding: utf-8 -*-
import os
import sys
from os import path, listdir, mkdir
import numpy as np
import random
import tensorflow as tf

import timeit
import cv2
# import skimage.io
from tqdm import tqdm
np.random.seed(1)
tf.set_random_seed(1)
np.seterr(divide='ignore', invalid='ignore')

from pixel_decoder.utils import stats_data, open_image, preprocess_inputs_std, cache_stats
from pixel_decoder.resnet_unet import get_resnet_unet

def predict(imgs_folder, test_folder, models_folder, pred_folder, origin_shape_no, border_no, channel_no, model_id):
    origin_shape = (origin_shape_no, origin_shape_no)
    rgb_index = [0, 1, 2]
    border = (border_no, border_no)
    input_shape = (origin_shape[0] + border[0] + border[1] , origin_shape[1] + border[0] + border[1])
    means, stds = cache_stats(imgs_folder)
    if not path.isdir(pred_folder):mkdir(os.path.join(os.getcwd(),pred_folder))
    if not path.isdir(path.join(pred_folder, model_id)):mkdir(path.join(pred_folder, model_id))
    if model_id == 'resnet_unet':
        model = get_resnet_unet(input_shape, channel_no)
    else:
        from inception_unet import get_inception_resnet_v2_unet
        model = get_inception_resnet_v2_unet(input_shape, channel_no)
    model.load_weights(path.join(models_folder, '{}_weights4.h5'.format(model_id)))

    if not path.isdir(models_folder):
        mkdir(models_folder)

    for f in tqdm(sorted(listdir(path.join(test_folder)))):
        if path.isfile(path.join(test_folder, f)) and '.tif' in f:
            img_id = f.split('.')[0]

            fpath = path.join(test_folder, f)
            img = open_image(fpath)
            # if img.shape[0] != origin_shape[0]:
            #     img= cv2.resize(img, origin_shape)
            # else:img = img
            if channel_no == 8:img = img
            else:
                band_index = rgb_index
                img = img[:, :, band_index]
            img = cv2.copyMakeBorder(img, border[0], border[1], border[0], border[1], cv2.BORDER_REFLECT_101)
            inp = []
            inp.append(img)
            inp.append(np.rot90(img, k=1))
            inp = np.asarray(inp)
            inp = preprocess_inputs_std(inp, means, stds)
            pred = model.predict(inp)
            mask = pred[0] + np.rot90(pred[1], k=3)
            mask /= 2
            mask_index1 = border[0]
            mask_index2 = input_shape[1] - border[1]
            mask = mask[mask_index1:mask_index2, mask_index1:mask_index2, ...]
            mask = mask * 255
            mask = mask.astype('uint8')
            cv2.imwrite(path.join(pred_folder, model_id,'{}.png'.format(img_id)), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
