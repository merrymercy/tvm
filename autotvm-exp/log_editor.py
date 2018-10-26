"""
Show stats for a log record file
"""
import sys
import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import tvm
import topi
from tvm import autotvm
from tvm.autotvm.record import load_from_file, encode
from tvm.autotvm.task.nnvm_integration import TaskExtractEnv

def get_wkl_partition(filename):
    '''partition according to workloads'''
    wkls = OrderedDict()
    for inp, res in load_from_file(filename):
        wkl = inp.task.workload
    
        if wkl not in wkls:
            # calc flops
            tsk = autotvm.task.create(inp.task.name, inp.task.args,
                                      inp.target,
                                      template_key=inp.config.template_key)
    
            names = list(tsk.config_space.space_map.keys())
            dims = [len(x) for x in tsk.config_space.space_map.values()]
            wkls[wkl] = {"total_flop": tsk.flop, "records": [],
                         "names": names, "dims": dims,
                         "space": tsk.config_space,
                         "workload": wkl}

        wkls[wkl]["records"].append((inp, res))

    return wkls

def print_basic(wkl):
    """print basic information"""
    total_flop = wkl["total_flop"]
    flopses = []
    for inp, res in wkl["records"]:
        flopses.append(total_flop / np.mean(res.costs) / 1e9 if res.error_no == 0 else 0)
    flopses = np.array(flopses)

    # plot distribution
    print("valid/total: %d/%d" % (np.sum(flopses > 1e-5), len(flopses)))
    print("max, min, mean: %.2f, %.2f, %.2f" % (np.max(flopses), np.min(flopses), np.mean(flopses)))

def plot_flop_distribution(wkl):
    total_flop = wkl['total_flop']
    flopses = []
    for inp, res in wkl["records"]:
        flopses.append(total_flop / np.mean(res.costs) / 1e9 if res.error_no == 0 else 0)
    flopses = np.array(flopses)
    plt.hist(flopses, bins=200)
    plt.ylabel("number")
    plt.xlabel("GFLOPS")
    plt.title(wkl['workload'])
    plt.show()

def print_variance(wkl):
    """detect the variance of measurement"""
    total_flop = wkl['total_flop']
    idx2flops = {}

    for inp, res in wkl['records']:
        idx = inp.config.index
        if idx not in idx2flops:
            idx2flops[idx] = []

        idx2flops[idx].append(total_flop / np.mean(res.costs) / 1e9 if res.error_no == 0 else 0)

    for k, v in idx2flops.items():
        if len(v) > 1:
            print("duplication:", v)

def save_to_file(wkl, filename):
    with open(filename, 'w') as fout:
        for inp, res in wkl['records']:
            fout.write(encode(inp, res) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, help="Input filename")
    parser.add_argument("--i2", type=str, help="Input filename")
    parser.add_argument("--var", action='store_true', help='show variance of measurement results')
    parser.add_argument("--save", action='store_true', help='save to file')
    args = parser.parse_args()

    # init TOPI tuning tasks
    TaskExtractEnv.get()

    # partition
    wkls = get_wkl_partition(args.i)

    # plot 
    print("number of workloads:", len(wkls))

    for i, wkl in enumerate(wkls.values()):
        print(i, wkl['workload'])

        print_basic(wkl)
        if args.var:
            print_variance(wkl)

        if args.save:
            save_to_file(wkl, "%s.%d" % (args.i, i))

        print()

        #plot_flop_distribution(wkl)

