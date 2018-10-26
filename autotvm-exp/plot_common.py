import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

import numpy as np
from util import query_log_key, query_flop, enhance_color

in_file = 'data/all_exp.log'
output = 'figures/tuners.pdf'
device = 'GeForce GTX TITAN X'
task_names = [
    'resnet.C1.B1', 'resnet.C2.B1', 'resnet.C3.B1',
    'resnet.C4.B1', 'resnet.C5.B1', 'resnet.C6.B1',
    'resnet.C7.B1', 'resnet.C8.B1', 'resnet.C9.B1',
    'resnet.C10.B1', 'resnet.C11.B1', 'resnet.C12.B1',
]

select_task_names = [
    'resnet.C1.B1', 'resnet.C2.B1', 'resnet.C5.B1', 'resnet.C6.B1',
]

def draw(task_names, methods, output, show_name, args, x_max=None, col=4, yerr_max=1e9,
         legend_ax=None, method2color=None, offset=None, add_cap=False):
    COL = col

    n_tasks = len(task_names)
    fig, axes = plt.subplots((n_tasks + COL-1) // COL, COL)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    fontsize = 10
    if legend_ax is None:
        legend_ax = min(1, len(task_names)-1)
    objs = []
    all_legends = []

    for k, task_name in enumerate(task_names):

        x_max_backup = x_max
        if 'B1.i' in task_name:
            x_max = 800

        all_curves = []
        all_colors = []
        all_methods = []
        ax = axes[k]
        for method in methods:
            raw_method = method
            if '*' in method:
                method, step = method.split('*')
                step = int(step)
            else:
                step = 1

            if 'spatial_pack' in method:
                value = query_log_key('llvm', 'rpi3b', task_name, method, in_file)
            else:
                value = query_log_key('cuda', device, task_name, method, in_file)

            if value is None:
                print("cannot find", task_name, method)
                continue

            sum_curve = np.array(eval(value))

            flop = query_flop(task_name)
            sum_curve = flop / sum_curve / 1e12

            keep = 0
            for i in range(len(sum_curve)):
                for j in range(len(sum_curve[i])):
                    keep = max(keep, sum_curve[i][j])
                    sum_curve[i][j] = keep

            y = np.mean(sum_curve, axis=0)
            y_err = np.std(sum_curve, axis=0) * 0.5

            if x_max is None:
                x_max = len(y)

            y = y[:x_max*step:step]
            y_err = y_err[:x_max*step:step]
            y_err = np.where(y_err < yerr_max, y_err, np.ones_like(y_err) * yerr_max)

            all_curves.append((y, y_err))
            all_methods.append(raw_method)

            if k == legend_ax:
                all_legends.append(show_name(raw_method))

            if method2color is not None:
                all_colors.append(method2color(raw_method))

        x = np.arange(1, x_max + 1)
        for i, (y, y_err) in enumerate(all_curves):
            if all_colors:
                obj = ax.plot(x, y, color=all_colors[i])
                ax.fill_between(x, y - y_err, y + y_err, color=all_colors[i], alpha=0.15)
            else:
                obj = ax.plot(x, y)
                ax.fill_between(x, y - y_err, y + y_err, alpha=0.15)

            if k == legend_ax:
                objs.append(obj[0])

        ax.set_title(show_name(task_name.split('.')[1]))
        ax.set_xlim([1, x_max])
        if k // COL == (len(task_names)-1) // COL:
            if add_cap:
                ax.set_xlabel("Number of Trials\n("  +  ("abcd"[k]) + ')')
            else:
                ax.set_xlabel("Number of Trials")
        if k % COL == 0:
            ax.set_ylabel("TFLOPS")

        if x_max_backup is not None:
            x_max = x_max_backup

    if add_cap:
        delta = 0.20
    else:
        delta = 0
    fig.set_size_inches(10, 2.5 * len(task_names) // COL + delta)

    if args.full:
        offset = offset or (1.2, 1.35)
        axes[legend_ax].legend(objs, all_legends,
                               loc='upper center', bbox_to_anchor=offset,
                               ncol=len(objs))
    else:
        offset = offset or (1.2, 1.58)
        if len(task_names) == 3:
            offset = (0.50, 1.58)
        axes[legend_ax].legend(objs, all_legends,
                               ncol=len(objs),
                               loc='upper center', bbox_to_anchor=offset)

    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    if not args.s:
        plt.show()
    plt.close()

    print("Output to %s" % output)

