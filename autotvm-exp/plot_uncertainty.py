import argparse
from plot_common import *

methods = [
    'xgb-reg-ei', 'xgb-reg-ucb', 'xgb-reg-mean',
]

def show_name(name):
    trans_table = {
        'xgb-reg-ei': 'Expected Improvement',
        'xgb-reg-ucb': 'Upper Confidence Bound',
        'xgb-reg-mean': 'Mean',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = 'figures/uncertainty_full.pdf'
    else:
        output = 'figures/uncertainty.pdf'
        task_names = select_task_names

    draw(task_names, methods, output, show_name, args, x_max=800, col=4)

