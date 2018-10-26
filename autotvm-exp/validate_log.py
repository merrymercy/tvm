"""Show the variance of measurement in log records"""

import argparse

import numpy as np

import tvm
import topi
from tvm import autotvm

from tvm.autotvm.task.nnvm_integration import TaskExtractEnv

TaskExtractEnv.get()

wkls = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, nargs="+")
    args = parser.parse_args()

    # parse logs
    for filename in args.i:
        for inp, res in autotvm.record.load_from_file(filename):
            wkl = inp.task.workload 

            if wkl not in wkls:
                # calc flops
                tsk = autotvm.task.create(inp.task.name, inp.task.args,
                                          inp.target,
                                          template_key=inp.config.template_key)
        
                names = list(tsk.config_space.space_map.keys())
                dims = [len(x) for x in tsk.config_space.space_map.values()]
                wkls[wkl] = {"total_flop": tsk.flop, "records": {},
                             "names": names, "dims": dims,
                             "space": tsk.config_space,
                             "workload": wkl}

            if inp.config.index not in wkls[wkl]['records']:
                wkls[wkl]['records'][inp.config.index] = []

            flop = wkls[wkl]['total_flop'] / np.mean(res.costs) / 1e9 if res.error_no == 0 else 0

            if flop > 1:
                wkls[wkl]['records'][inp.config.index].append((filename, np.mean(res.costs)))

    # report
    min_differ_costs = 1e9
    points = {}

    for spaces in wkls.values():
        for conflict_set in spaces['records'].values():
            if len(conflict_set) >= 2:
                values = [x[1] for x in conflict_set]
                if np.std(values) / np.mean(values) > 0.04:
                    conflict_set.sort(key=lambda x:x[1])

                    win = conflict_set[0][0]
                    find_other = False
                    for i in range(1, len(conflict_set)):
                        if conflict_set[i][0] != win:
                            find_other = True

                    if find_other:
                        if win not in points:
                            points[win] = 0
                        points[win] += 1
                    else:
                        continue

                    min_differ_costs = min(min_differ_costs, np.min(values))
                    for item in conflict_set:
                        print(item)
                    print()

    print("Min differ cost: ", min_differ_costs)
    print("Points: ", points)

