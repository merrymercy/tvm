import argparse
from plot_common import *

methods = [
    'xgb-rank', 'treegru-rank',
    'xgb-reg', 'treegru-reg',
]

def show_name(name):
    trans_table = {
        'xgb-rank': 'GBT Rank',
        'treegru-rank': 'TreeGRU Rank',

        'xgb-reg': 'GBT Regression',
        'treegru-reg': 'TreeGRU Regression',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = 'figures/rank_reg_full.pdf'
    else:
        output = 'figures/rank_reg.pdf'
        task_names = select_task_names

    draw(task_names, methods, output, show_name, args, x_max=800, col=4)

