import argparse
from plot_common import *

methods = [
    'xgb-rank', 'treernn-rank',
    'ga', 'ga*3',
    'random', 'random*3',
]

def show_name(name):
    trans_table = {
        'xgb-rank': 'GBT',
        'treernn-rank': 'TreeGRU',
        'random': 'Random',
        'random*3': 'Random X 3',
        'ga': 'GA',
        'ga*3': 'GA X 3',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = 'figures/tuners_full.pdf'
    else:
        output = 'figures/tuners.pdf'
        task_names = task_names[:4]

    draw(task_names, methods, output, show_name, args, col=4)

