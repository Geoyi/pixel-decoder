# import os
from os import path, mkdir
from pixel_decoder.utils import datafiles, cache_stats, batch_data_generator, val_data_generator

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
from sklearn.model_selection import KFold
# import cv2
from keras.optimizers import SGD, Adam
from keras import metrics
from keras.callbacks import ModelCheckpoint
from pixel_decoder.loss import dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded, dice_logloss
from pixel_decoder.resnet_unet import get_resnet_unet
import keras.backend as K

def train(batch_size, imgs_folder, masks_folder, models_folder, model_id, origin_shape_no, border_no, classes =1, channel_no=3):
    origin_shape = (int(origin_shape_no), int(origin_shape_no))
    border = (int(border_no), int(border_no))
    input_shape = origin_shape
    all_files, all_masks = datafiles(imgs_folder, masks_folder)
    means, stds = cache_stats(imgs_folder)
    if model_id == 'resnet_unet':
        model = get_resnet_unet(input_shape, channel_no, classes)
    else:
        print('No model loaded!')

    if not path.isdir(models_folder):
        mkdir(models_folder)

    kf = KFold(n_splits=4, shuffle=True, random_state=1)
    for all_train_idx, all_val_idx in kf.split(all_files):
        train_idx = []
        val_idx = []

        for i in all_train_idx:
            train_idx.append(i)
        for i in all_val_idx:
            val_idx.append(i)

        validation_steps = int(len(val_idx) / batch_size)
        steps_per_epoch = int(len(train_idx) / batch_size)

        if validation_steps == 0 or steps_per_epoch == 0:
          continue

        print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

        np.random.seed(11)
        random.seed(11)
        tf.set_random_seed(11)
        print(model.summary())
        batch_data_generat = batch_data_generator(train_idx, batch_size, means, stds, imgs_folder, masks_folder, models_folder, channel_no, border_no, origin_shape_no)
        val_data_generat = val_data_generator(val_idx, batch_size, validation_steps, means, stds, imgs_folder, masks_folder, models_folder, channel_no, border_no, origin_shape_no)


        model.compile(loss=dice_logloss3,
                    optimizer=SGD(lr=5e-2, decay=1e-6, momentum=0.9, nesterov=True),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

        model_checkpoint = ModelCheckpoint(path.join(models_folder, '{}_weights.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generat,
                            epochs=25, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
        for l in model.layers:
          l.trainable = True
        model.compile(loss=dice_logloss3,
                    optimizer=Adam(lr=1e-3),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

        model.fit_generator(generator=batch_data_generat,
                            epochs=40, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                             validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
        model.optimizer = Adam(lr=2e-4)
        model.fit_generator(generator=batch_data_generat,
                            epochs=25, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])

        np.random.seed(22)
        random.seed(22)
        tf.set_random_seed(22)
        model.load_weights(path.join(models_folder, '{}_weights.h5'.format(model_id)))
        model.compile(loss=dice_logloss,
                    optimizer=Adam(lr=5e-4),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        model_checkpoint2 = ModelCheckpoint(path.join(models_folder, '{}_weights2.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generat,
                            epochs=30, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint2])
        optimizer=Adam(lr=1e-5)
        model.fit_generator(generator=batch_data_generat,
                            epochs=20, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint2])

        np.random.seed(33)
        random.seed(33)
        tf.set_random_seed(33)
        model.load_weights(path.join(models_folder, '{}_weights2.h5'.format(model_id)))
        model.compile(loss=dice_logloss2,
                    optimizer=Adam(lr=5e-5),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        model_checkpoint3 = ModelCheckpoint(path.join(models_folder, '{}_weights3.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generat,
                            epochs=50, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint3])

        np.random.seed(44)
        random.seed(44)
        tf.set_random_seed(44)
        model.load_weights(path.join(models_folder, '{}_weights3.h5'.format(model_id)))
        model.compile(loss=dice_logloss3,
                    optimizer=Adam(lr=2e-5),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        model_checkpoint4 = ModelCheckpoint(path.join(models_folder, '{}_weights4.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generat,
                            epochs=50, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint4])
        K.clear_session()
