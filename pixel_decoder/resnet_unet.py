# from keras.applications.vgg16 import VGG16
# from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Activation
from pixel_decoder.resnet_back import ResNet50, identity_block
from pixel_decoder.resnet_back import conv_block as resnet_conv_block
# from keras.losses import binary_crossentropy
# from loss import dice_coef, dice_coef_rounded, dice_coef_loss, dice_logloss, dice_logloss2, dice_logloss3, weighted_bce_loss, weighted_bce_dice_loss
# bn_axis = 3
# channel_axis = bn_axis

def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    bn_axis = 3
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = BatchNormalization(name=name, axis=bn_axis)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv

def get_resnet_unet(input_shape,channel_no, classes = 1, weights='imagenet'):
    bn_axis = 3
    inp = Input(input_shape + (channel_no,))

    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inp)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    conv1 = x
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    enc1 = x

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    enc2 = x

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    enc3 = x

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    enc4 = x

    up6 = concatenate([UpSampling2D()(enc4), enc3], axis=-1)
    conv6 = conv_block(up6, 128)
    conv6 = conv_block(conv6, 128)

    up7 = concatenate([UpSampling2D()(conv6), enc2], axis=-1)
    conv7 = conv_block(up7, 96)
    conv7 = conv_block(conv7, 96)

    up8 = concatenate([UpSampling2D()(conv7), enc1], axis=-1)
    conv8 = conv_block(up8, 64)
    conv8 = conv_block(conv8, 64)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block(up9, 48)
    conv9 = conv_block(conv9, 48)

    up10 = concatenate([UpSampling2D()(conv9), inp], axis=-1)
    conv10 = conv_block(up10, 32)
    conv10 = conv_block(conv10, 32)
    res = Conv2D(classes, (1, 1), activation='sigmoid')(conv10)
    model = Model(inp, res)

    if weights == 'imagenet':
        resnet = ResNet50(input_shape=input_shape + (3,), include_top=False, weights=weights)
        for i in range(2, len(resnet.layers)-1):
            model.layers[i].set_weights(resnet.layers[i].get_weights())
            model.layers[i].trainable = False

    return model
