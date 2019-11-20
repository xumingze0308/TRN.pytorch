import argparse

__all__ = ['parse_base_args']

def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=21, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=5e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=25, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    return parser
