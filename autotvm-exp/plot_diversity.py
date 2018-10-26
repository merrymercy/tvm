import argparse
from plot_common import *

methods = [
    'xgb-rank', 'xgb-rank-d2', 'xgb-rank-d4',
]

def show_name(name):
    trans_table = {
        'xgb-rank': '$\lambda$=1',
        'xgb-rank-d2': '$\lambda$=2',
        'xgb-rank-d4': '$\lambda$=4',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = 'figures/diversity_full.pdf'
    else:
        output = 'figures/diversity.pdf'
        task_names = select_task_names

    draw(task_names, methods, output, show_name, args, x_max=800, col=4)

