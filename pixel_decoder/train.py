import os
# import sys
from os import path, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
from sklearn.model_selection import KFold
import cv2
from keras.optimizers import SGD, Adam
from keras import metrics
from keras.callbacks import ModelCheckpoint
from pixel_decoder.loss import dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded, dice_logloss
import skimage.io
import keras.backend as K
from pixel_decoder.utils import datafiles, stats_data, cache_stats, preprocess_inputs_std, rotate_image, batch_data_generator, val_data_generator


# channel_no = 3
# input_shape = (320, 320)
# batch_size = 16
# model_id = sys.argv[1]
# imgs_folder = sys.argv[2]
# masks_folder = sys.argv[3]
# models_folder =sys.argv[4]

def model(batch_size, input_shape, channel_no, imgs_folder, masks_folder,models_folder, model_id='resnet_unet'):
    all_files, all_masks = datafiles()
    means, stds = cache_stats(imgs_folder)
    if model_id == 'resnet_unet':
        from pixel_decoder.resnet_unet import get_resnet_unet
        model = get_resnet_unet(input_shape, channel_no)
    # elif model_id == 'inception_unet':
    #     from pixel_decoder.inception_unet import get_inception_resnet_v2_unet
    #     model = get_inception_resnet_v2_unet(input_shape, channel_no)
    # elif model_id == 'linknet_unet':
    #     from pixel_decoder.linknet_unet import get_resnet50_linknet
    #     model = get_resnet50_linknet(input_shape, channel_no)
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


        model.compile(loss=dice_logloss3,
                    optimizer=SGD(lr=5e-2, decay=1e-6, momentum=0.9, nesterov=True),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

        model_checkpoint = ModelCheckpoint(path.join(models_folder, '{}_weights.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generator(train_idx, batch_size, means, stds),
                            epochs=25, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps, means, stds),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
        for l in model.layers:
          l.trainable = True
        model.compile(loss=dice_logloss3,
                    optimizer=Adam(lr=1e-3),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

        model.fit_generator(generator=batch_data_generator(train_idx, batch_size, means, stds),
                            epochs=40, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps, means, stds),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
        model.optimizer = Adam(lr=2e-4)
        model.fit_generator(generator=batch_data_generator(train_idx, batch_size, means, stds),
                            epochs=25, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps, means, stds),
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
        model.fit_generator(generator=batch_data_generator(train_idx, batch_size, means, stds),
                            epochs=30, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps, means, stds),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint2])
        optimizer=Adam(lr=1e-5)
        model.fit_generator(generator=batch_data_generator(train_idx, batch_size, means, stds),
                            epochs=20, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps, means, stds),
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
        model.fit_generator(generator=batch_data_generator(train_idx, batch_size, means, stds),
                            epochs=50, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps, means, stds),
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
        model.fit_generator(generator=batch_data_generator(train_idx, batch_size, means, stds),
                            epochs=50, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps, means, stds),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint4])
        K.clear_session()
