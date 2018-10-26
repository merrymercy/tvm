"""Compare tuners in single operator tuning tasks"""

import argparse
import logging
import time
import os
import pickle
import uuid
import json

import numpy as np

import nnvm.testing
import nnvm.compiler
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, TreeGRUTuner
from tvm.autotvm.task.nnvm_integration import TaskExtractEnv
from tvm.contrib import util
import tvm.contrib.graph_runtime as runtime

from util import log_to_dashboard, workload_to_name, log_value, array2str_round

TRACKER_HOST = 'localhost'
TRACKER_PORT = 9190

def get_device_name(rpc_key, target):
    """Get the name of a remote device"""
    remote = autotvm.measure.request_remote(device_key, TRACKER_HOST, TRACKER_PORT)
    ctx = remote.context(str(target), 0)
    return ctx.device_name or device_key


def tune_tasks(tasks,
               measure_option,
               device_key,
               tuner,
               n_trial,
               early_stopping,
               cache_filename,
               try_winograd,
               n_repeat):
    """ tune a list of tasks """
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception as e:
                pass

    # create tmp files
    tmp_cache_file = cache_filename + ".tmp"
    if os.path.exists(tmp_cache_file):
        #os.remove(tmp_cache_file)
        pass

    # get device name
    target = tasks[0].target
    device_name = get_device_name(device_key, target)

    # tune tasks sequentially
    for i, tsk in enumerate(tasks):
        if args.start_from and i+1 < args.start_from:
            continue

        time_costs = []
        time_stamps = []
        for j in range(n_repeat):
            prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

            # create tuner
            if tuner == 'xgb-rank':
                tuner_obj = XGBTuner(tsk, loss_type='rank')
            elif tuner == 'xgb-rank-d2':
                tuner_obj = XGBTuner(tsk, loss_type='rank', diversity_filter_ratio=2)
            elif tuner == 'xgb-rank-d4':
                tuner_obj = XGBTuner(tsk, loss_type='rank', diversity_filter_ratio=4)
            elif tuner == 'xgb-rank-no-eps':
                tuner_obj = XGBTuner(tsk, loss_type='rank', eps_greedy=0)
            elif tuner == 'xgb-reg':
                tuner_obj = XGBTuner(tsk, loss_type='reg')
            elif tuner == 'xgb-reg-mean':
                tuner_obj = XGBTuner(tsk, loss_type='reg', acq_type='mean')
            elif tuner == 'xgb-reg-ei':
                tuner_obj = XGBTuner(tsk, loss_type='reg', acq_type='ei')
            elif tuner == 'xgb-reg-ucb':
                tuner_obj = XGBTuner(tsk, loss_type='reg', acq_type='ucb')
            elif tuner == 'treegru-rank':
                tuner_obj = TreeGRUTuner(tsk, loss_type='rank')
            elif tuner == 'treegru-reg':
                tuner_obj = TreeGRUTuner(tsk, loss_type='reg')
            elif tuner == 'ga':
                tuner_obj = GATuner(tsk, pop_size=100)
            elif tuner == 'random':
                tuner_obj = RandomTuner(tsk)
            elif tuner == 'gridsearch':
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            monitor = autotvm.callback.Monitor()

            callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix),
                       autotvm.callback.log_to_file(tmp_cache_file),
                       monitor]

            if args.dashboard:
                if args.metajob_name:
                    meta_job_uuid = 'abcdefg'
                else:
                    meta_job_uuid = 'fffffff'

                callbacks.append(log_to_dashboard(
                    meta_job_uuid,
                    args.metajob_name or 'tmp',
                    workload_to_name(tsk.workload),
                    {'tuner': tuner,
                     'n_trial': n_trial,
                     'early_stopping': early_stopping,
                     'try_winograd': try_winograd},
                     device_key))

            # do tuning
            tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                           early_stopping=early_stopping,
                           measure_option=measure_option,
                           callbacks=callbacks)

            time_costs.append(monitor.trial_costs())
            time_stamps.append(monitor.trial_timestamps())

            del callbacks

        # log experiment results to a single record file
        task_name = workload_to_name(tasks[i].workload).replace('resnet-18', 'resnet')

        value = array2str_round(time_costs)
        log_value(target.target_name,
                  device_name, task_name, tuner, value, outfile='data/all_exp.log')
        value = array2str_round(time_stamps, decimal=2)
        log_value(target.target_name,
                  device_name, task_name, tuner + '-time', value, outfile='data/all_exp.log')

    # pick best records to a cache file (used for compilation later)
    autotvm.record.pick_best(tmp_cache_file, cache_filename)


def get_target(target):
    # target device
    target_table = {
       'local':      ('local','cuda -model=titanx', None),

       'rpi3b-cpu':  ('rpi3b',
                      tvm.target.arm_cpu('rasp3b'), None),
       'pynq-cpu':      ('pynq',
                         tvm.target.arm_cpu('pynq'), None),
       'ultra96-cpu':   ('ultra96',
                         tvm.target.arm_cpu('ultra96'), None),

       'rk3399-cpu': ('rk3399',
                      tvm.target.arm_cpu('rk3399'), None),
       'rk3399-gpu': ('rk3399',
                      'opencl -model=rk3399 -device=mali', tvm.target.arm_cpu('rk3399')),
       'hikey960-cpu':  ('hikey960',
                         'llvm -model=hikey960 -device=arm_cpu -mtriple=aarch64-linux-gnu', None),
       'hikey960-gpu':  ('hikey960',
                         'opencl -model=hikey960 -device=mali', 'llvm -mtriple=aarch64-linux-gnu'),
       'mate10pro-cpu': ('mate10pro',
                         tvm.target.arm_cpu('mate10pro'), None),
       'mate10pro-gpu': ('mate10pro',
                         'opencl -model=mate10pro -device=mali', tvm.target.arm_cpu('mate10pro')),
       'p20pro-cpu':    ('p20pro',
                         tvm.target.arm_cpu('p20pro'), None),
       'p20pro-gpu':    ('p20pro',
                         'opencl -model=p20pro', tvm.target.arm_cpu('p20pro')),
       'pixel2-cpu':    ('pixel2',
                         tvm.target.arm_cpu('pixel2'), None),
       'pixel2-gpu':    ('pixel2',
                         'opencl -model=pixel2 -device=mali', tvm.target.arm_cpu('pixel2')),

       'mi6-cpu':       ('mi6',
                         'llvm -model=mi6 -device=arm_cpu -mtriple=arm64-linux-android', None),
       'mi6-gpu':       ('mi6',
                         'opencl -model=mi6 -device=mali', 'llvm -target=arm64-linux-android'),

       '1080ti':        ('1080ti', 'cuda -model=1080ti', None),
       'titanx':        ('titanx', 'cuda -model=titanx', None),
       'gfx900':        ('gfx900', 'rocm -model=gfx900', None),
       'tx2':           ('tx2', 'cuda -model=tx2', 'llvm -target=aarch64-linux-gnu'),
    }

    device_key, target, target_host = target_table[target]
    target = tvm.target.create(target)

    return device_key, target, target_host

def get_tuning_option(device_key, args):
    # extract tasks and tuning
    tuning_option = {
       'tuner': args.tuner,
       'device_key': device_key,

       'n_trial': args.n_trial,
       'cache_filename': args.cache_file,

       'try_winograd': args.winograd,
       'n_repeat': args.n_repeat,
    }

    table = {        # build_t  run_t  number  repeat  min_repeat_ms  n_times  early_stopping  build_func
       'local':        (10,     3,     10,     2,      0,             5,       400,            'default'),                                                   
       # arm cpu
       'pynq-cpu':     (10,     8,     2,      3,      100,           5,       400,            'default'),
       'ultra96-cpu':  (10,     6,     3,      3,      100,           5,       400,            'default'),
       'rpi3b-cpu':    (10,     6,     3,      3,      100,           10,      400,            'default'),
       'rk3399-cpu':   (10,     6,     3,      3,      100,           10,      400,            'default'),
       'hikey960-cpu': (10,     6,     3,      3,      100,           10,      400,            'default'),
       'p20pro-cpu':   (10,     6,     3,      3,      100,           10,      400,            'ndk'),
       'pixel2-cpu':   (10,     6,     3,      3,      100,           10,      400,            'ndk'),

       # mobile gpu
       'rk3399-gpu':   (10,     6,     4,      3,      100,           50,      400,            'default'),
       'hikey960-gpu': (10,     6,     4,      3,      100,           50,      400,            'default'),
       'p20pro-gpu':   (10,     6,     4,      3,      100,           50,      600,            'ndk'),
       'pixel2-gpu':   (10,     6,     4,      3,      100,           50,      400,            'ndk'),

       # nvidia gpu
       '1080ti':       (10,     5,     10,     3,      0,             500,     600,            'default'),
       'titanx':       (10,     5,     10,     3,      150,           500,     None,           'default'),

       # amd gpu
       'gfx900':       (10,     5,     10,     3,      150,           500,     600,            'default'),
       'tx2':          (10,     6,     5,      3,      100,           300,     600,            'default'),
    }

    table['mate10pro-cpu'] = table['p20pro-cpu']
    table['mate10pro-gpu'] = table['p20pro-gpu']

    build_t, run_t, number, repeat, min_repeat_ms, \
            n_times, early_stopping, build_func = table[args.target]

    tuning_option['early_stopping'] = early_stopping
    tuning_option['measure_option'] = autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func=build_func),
        runner=autotvm.RPCRunner(
            device_key,
            host=TRACKER_HOST, port=TRACKER_PORT,
            timeout=run_t,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            check_correctness=False)
    )

    return tuning_option, n_times


def get_network(name, batch_size, dtype='float32'):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if name == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet_v2':
        net, params = nnvm.testing.mobilenet_v2.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (batch_size, 3, 299, 299)
        net, params = nnvm.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "densenet" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.densenet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "squeezenet" in name:
        version = name.split("_v")[1]
        net, params = nnvm.testing.squeezenet.get_workload(batch_size=batch_size, version=version, dtype=dtype)
    elif name == 'custom':
        # an example for custom network
        from nnvm.testing import utils
        net = nnvm.sym.Variable('data')
        net = nnvm.sym.conv2d(net, channels=4, kernel_size=(3,3), padding=(1,1))
        net = nnvm.sym.flatten(net)
        net = nnvm.sym.dense(net, units=1000)
        net, params = utils.create_workload(net, batch_size, (3, 224, 224), dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        net, params = nnvm.frontend.from_mxnet(block)
        net = nnvm.sym.softmax(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape

def set_cuda_arch(device_key):
    from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
    remote = autotvm.measure.request_remote(device_key, TRACKER_HOST, TRACKER_PORT)
    ctx = remote.gpu(0)
    set_cuda_target_arch('sm_' + ctx.compute_version.replace('.', ''))
    del ctx
    del remote

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default='resnet-18')
    parser.add_argument("--target", type=str, default='titanx')
    parser.add_argument("--target-host", type=str)
    parser.add_argument("--n-trial", type=int, default=10)
    parser.add_argument("--n-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0x93)
    parser.add_argument("--cache-file", type=str)
    parser.add_argument("--mode", type=str, default='tune')
    parser.add_argument("--check", action='store_true')
    parser.add_argument("--winograd", action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--dtype", type=str, default='float32')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument("--metajob-name", type=str)
    parser.add_argument("--start-from", type=int)
    parser.add_argument("--tuner", type=str, default='xgb-rank')
    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.debug:
        import logging
        logging.basicConfig()
        logger = logging.getLogger("autotvm")
        logger.setLevel(logging.DEBUG)

    dtype = args.dtype

    args.cache_file = args.cache_file or \
            args.network + "." + args.target + "." + args.tuner + "." + dtype + ".log"

    # device related
    device_key, target, target_host = get_target(args.target)
    tuning_option, n_times = get_tuning_option(device_key, args)

    # network
    net, params, input_shape, out_shape = get_network(args.network, batch_size=1, dtype=dtype)
    if args.mode =='tune': 
        symbols = (nnvm.sym.conv2d, nnvm.sym.conv2d_transpose)
        if 'mali' in str(target):
            symbols += (nnvm.sym.dense,)
        tasks = autotvm.task.extract_from_graph(net, shape={'data': input_shape}, dtype=dtype,
                                                symbols=symbols,
                                                target=target, target_host=target_host)
        tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    with autotvm.apply_history_best(args.cache_file):
        raw_params = params

        if 'cuda' in str(target):
            set_cuda_arch(device_key)

        print("Compile...")
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(
                net, target=target, target_host=target_host,
                shape={'data': input_shape}, params=params, dtype=dtype)

        tmp = util.tempdir()
        if 'android' in str(target) or 'android' in str(target_host):
            from tvm.contrib import ndk
            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        print("Upload...")
        if device_key =='local':
            ctx = tvm.context(str(target), 0)
            rlib = lib
        else:
            remote = autotvm.measure.request_remote(device_key, TRACKER_HOST, TRACKER_PORT,
                                                    timeout=10000)
            remote.upload(tmp.relpath(filename))
            ctx = remote.context(str(target), 0)
            rlib = remote.load_module(filename)

        # upload parameters
        module = runtime.create(graph, rlib, ctx)

        data_tvm = tvm.nd.array((10 * np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        module.run()
        output = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx, dtype=dtype)).asnumpy()

        # check correctness
        if args.check:
            print("Check...")
            with nnvm.compiler.build_config():
                graph, lib, params = nnvm.compiler.build(
                    net, target='llvm',
                    shape={'data': input_shape}, params=raw_params, dtype=dtype)

            ref_ctx = tvm.cpu()
            ref_module = runtime.create(graph, lib, ref_ctx)
            ref_module.set_input('data', data_tvm)
            ref_module.set_input(**params)
            ref_module.run()
            out_reference = ref_module.get_output(0,
                    tvm.nd.empty(out_shape, ctx=ref_ctx, dtype=dtype)).asnumpy()

            print(output.flatten()[:10])
            np.testing.assert_allclose(out_reference, output, rtol=1e-2)

        # evaluate
        print("Evaluate...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=n_times)
        prof_res = ftimer()
        print("\n" + args.network + " " + args.target + " " + str(prof_res.mean), "\n")

