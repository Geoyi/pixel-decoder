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
    pparser.add_argument('-m', '--out_model', default='out_model', type=str,
                     help='directory for storing output files')

    subparsers = parser0.add_subparsers(dest='command')

    parser = subparsers.add_parser('train', parents=[pparser], help='train the model', formatter_class=dhf)
    parser.add_argument('-bz', '--batch_size', help='batch size for the training', required=True)
    parser.add_argument('-imgs', '--imgs_folder', help='directory for RGB images to train', required=True)
    parser.add_argument('-masks', '--masks_folder', help='directory for labeled mask to train', required=True)
    # parser.add_argument('-out', '--out_model', help='directory for labeled mask to train', required=True)
    parser.add_argument('-mid', '--model_id', help='model id from README', default='resnet_unet', required=False)
    parser.add_argument('-shape', '--origin_shape', help='the image shape of the input training images', default=(256, 256), required=False)
    parser.add_argument('-border', '--border', help='pixels number to add on training image and mask to get rid of edge effects from unet', default=(32, 32), required=False)
    parser.add_argument('-chns', '--channel_no', help='RGB three color channels for the training', default=3, type=int, required=False)


    # train(batch_size, imgs_folder, masks_folder, models_folder, model_id='resnet_unet', origin_shape=(256, 256), border=(32, 32), channel_no = 3)

    parser = subparsers.add_parser('predict', parents=[pparser], help='predict with test data', formatter_class=dhf)
    parser.add_argument('-imgs', '--imgs_folder', help='directory for RGB images to train', required=True)
    parser.add_argument('-masks', '--masks_folder', help='directory for labeled mask to train', required=True)
    parser.add_argument('-chns', '--channel_no', help='RGB three color channels for the training', default=3, type=int, required=False)
    parser.add_argument('-pred', '--pred_folder', help='directory for images to make prediction',  required=True)
    parser.add_argument('-mid', '--model_id', help='model id from README', default='resnet_unet', required=False)
    parser.add_argument('-ors', '--origin_shape', help='the image shape of the input training images', default=(256, 256), required=False)
    parser.add_argument('-bord', '--border', help='pixels number to add on training image and mask to get rid of edge effects from unet', default=(32, 32), required=False)
    parser.add_argument('-chns', '--channel_no', help='RGB three color channels for the training', default=3, type=int, required=False)

#predict(origin_shape, border, imgs_folder, masks_folder, models_folder, pred_folder, channel_no=3, model_id='resnet_unet')
    # turn Namespace into dictinary
    parsed_args = vars(parser0.parse_args(args))

    return parsed_args


def cli():
    args = parse_args(sys.argv[1:])
    logger.setLevel(args.pop('log') * 10)
    cmd = args.pop('command')
    out_folder = args.get('out_model')

    # create destination directory to save the trained model
    if not op.isdir(out_folder):
        makedirs(out_folder)

    if cmd == 'train':
        train(**kwargs)
    elif cmd == 'predict':
        predict(**kwargs)

if __name__ == "__main__":
    cli()
