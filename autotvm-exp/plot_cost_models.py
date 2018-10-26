import argparse
from plot_common import *

methods = [
    'xgb-rank', 'treegru-rank',
    'ga', 'ga*2',
    'random', 'random*2',
]

def show_name(name):
    trans_table = {
        'xgb-rank': 'GBT',
        'treegru-rank': 'TreeGRU',
        'random': 'Random',
        'random*2': 'Random X 2',
        'random*3': 'Random X 3',
        'ga': 'GA',
        'ga*2': 'GA X 2',
        'ga*3': 'GA X 3',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = 'figures/cost_model_full.pdf'
    else:
        output = 'figures/cost_model.pdf'
        task_names = select_task_names

    draw(task_names, methods, output, show_name, args, x_max=800, col=4)

