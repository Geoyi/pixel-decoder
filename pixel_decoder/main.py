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
    

    # command 2
    parser = subparsers.add_parser('predict', parents=[pparser], help='predict with test data', formatter_class=dhf)
    # parser.add_argument()

    # turn Namespace into dictinary
    parsed_args = vars(parser0.parse_args(args))

    return parsed_args


def cli():
    args = parse_args(sys.argv[1:])
    logger.setLevel(args.pop('log') * 10)
    cmd = args.pop('command')

    if cmd == 'cmd1':
        print(cmd)
    elif cmd == 'cmd2':
        print(cmd)

if __name__ == "__main__":
    cli()
