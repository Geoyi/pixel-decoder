import os
import sys
import argparse
import logging
from os import makedirs, path as op


from pixel_decoder.version import __version__
from pixel_decoder.train import train
from pixel_decoder.predict import predict

logger = logging.getLogger(__name__)


def parse_args(args):
    desc = 'pixel_decoder (v%s)' % __version__
    dhf = argparse.ArgumentDefaultsHelpFormatter
    parser0 = argparse.ArgumentParser(description=desc)

    pparser = argparse.ArgumentParser(add_help=False)
    pparser.add_argument('--version', help='Print version and exit', action='version', version=__version__)
    pparser.add_argument('--log', default=2, type=int,
                         help='0:all, 1:debug, 2:info, 3:warning, 4:error, 5:critical')

    subparsers = parser0.add_subparsers(dest='command')

    parser = subparsers.add_parser('train', parents=[pparser], help='train the model', formatter_class=dhf)
    parser.add_argument('-bz', '--batch_size', help='batch size for the training', default=16, type=int, required=True)
    parser.add_argument('-imgs', '--imgs_folder', help='directory for RGB images to train', type=str, required=True)
    parser.add_argument('-masks', '--masks_folder', help='directory for labeled mask to train', type=str, required=True)
    parser.add_argument('-model', '--models_folder', default='trained_models', help='directory for storing output files', type=str, required=True)
    parser.add_argument('-mid', '--model_id', help='model id from README', default='resnet_unet', type=str, required=False)
    parser.add_argument('-ors', '--origin_shape_no', help='the image shape of the input training images', default=256, type=int, required=False)
    parser.add_argument('-border', '--border_no', help='pixels number to add on training image and mask to get rid of edge effects from unet', default=32, type=int, required=False)
    # parser.add_argument('-chns', '--channel_no', help='RGB three color channels for the training', default=3, type=int, required=True)


    # train(batch_size, imgs_folder, masks_folder, models_folder, model_id='resnet_unet', origin_shape=(256, 256), border=(32, 32), channel_no = 3)

    parser = subparsers.add_parser('predict', parents=[pparser], help='predict with test data', formatter_class=dhf)
    parser.add_argument('-test', '--test_folder', help='directory for RGB images to predict', required=True)
    parser.add_argument('-imgs', '--imgs_folder', help='directory for labeled mask to train', type=str, required=True)
    parser.add_argument('-model', '--models_folder', default='trained_models', help='directory for storing output files', type=str, required=True)
    parser.add_argument('-pred', '--pred_folder', help='directory to save the predicted images',  required=True)
    parser.add_argument('-mid', '--model_id', help='model id from README', default='resnet_unet',type=str, required=False)
    parser.add_argument('-ors', '--origin_shape_no', help='the image shape of the input training images', default=256, type=int, required=False)
    parser.add_argument('-border', '--border_no', help='pixels number to add on training image and mask to get rid of edge effects from unet', default=32, type=int, required=False)
    # parser.add_argument('-chns', '--channel_no', help='RGB three color channels for the training', default=3, type=int, required=False)
#imgs_folder, test_folder, models_folder, pred_folder, origin_shape_no, border_no, model_id, channel_no=3
    parsed_args = vars(parser0.parse_args(args))

    return parsed_args

def main(cmd, **kwargs):
    if cmd == 'train':
        train(**kwargs)
    elif cmd == 'predict':
        predict(**kwargs)
# def main(cmd, batch_size, imgs_folder, test_folder, masks_folder, models_folder,pred_folder, model_id, origin_shape_no, border_no, channel_no):
#     if cmd == 'train':
#         train(batch_size, imgs_folder, masks_folder, models_folder, model_id, origin_shape_no, border_no, channel_no)
#     elif cmd == 'predict':
#         predict(imgs_folder, test_folder, models_folder, pred_folder, origin_shape_no, border_no, channel_no, model_id)


def cli():
    args = parse_args(sys.argv[1:])
    logger.setLevel(args.pop('log') * 10)
    # cmd = args.pop('command')
    # out_folder = args.get('trained_model')
    #
    # # create destination directory to save the trained model
    # if not op.isdir(op.join(os.getcwd(),out_folder)):
    #     makedirs(out_folder)
    main(args.pop('command'), **args)


if __name__ == "__main__":
    cli()
