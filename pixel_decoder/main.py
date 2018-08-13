import sys
import argparse
import logging
from pixel_decoder.version import __version__


logger = logging.getLogger(__name__)


def parse_args(args):
    desc = 'PACKAGENAME (v%s)' % __version__
    dhf = argparse.ArgumentDefaultsHelpFormatter
    parser0 = argparse.ArgumentParser(description=desc)

    pparser = argparse.ArgumentParser(add_help=False)
    pparser.add_argument('--version', help='Print version and exit', action='version', version=__version__)
    pparser.add_argument('--log', default=2, type=int,
                         help='0:all, 1:debug, 2:info, 3:warning, 4:error, 5:critical')

    # add subcommands
    subparsers = parser0.add_subparsers(dest='command')

    # command 1
    parser = subparsers.add_parser('cmd1', parents=[pparser], help='Command 1', formatter_class=dhf)
    # parser.add_argument()

    # command 2
    parser = subparsers.add_parser('cmd2', parents=[pparser], help='Command 2', formatter_class=dhf)
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
