import json
import os
import time
import uuid
from random import getrandbits

import numpy as np

import tvm
import topi
from tvm import autotvm


def log_value(target, device, task_name, method, value, outfile='tmp.log'):
    """
    append experiment result to a central log file
    Parameters
    ----------
    target: str
        one of 'cuda', 'opencl', 'llvm'
    device: str
        return string by TVMContext.device_name
    task_name: str
    method: str
    value: str
    outfile: str
    """

    with open(outfile, 'a') as fout:
        fout.write("\t".join([str(x) for x in
            (target, device, task_name, method, value, time.time())]) + '\n')


def array2str_round(x, decimal=6):
    """ print an array of float number to pretty string with round

    Parameters
    ----------
    x: Array of float or float
    decimal: int
    """
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return "[" + ", ".join([array2str_round(y, decimal=decimal)
                                for y in x]) + "]"
    format_str = "%%.%df" % decimal
    return format_str % x


def query_log_key(target, device, task_name, method, filename='all.log'):
    """ query value from uniform experiment log file
    the records in file should be logged by autotvm-exp/util/util.py log_value

    Parameters
    ----------
    target: str
        one of 'cuda', 'opencl', 'llvm'
    device: str
        return string by TVMContext.device_name
    task_name: str
    method: str
    filename: str
    """
    finds = []
    wanted = ''.join((target, device, task_name, method))
    with open(filename) as fin:
        for line in fin.readlines():
            items = line.split('\t')
            if len(items) != 6:
                continue
            target, device, task_name, method, value, tstamp = items
            key = ''.join((target, device, task_name, method))

            if key == wanted:
                finds.append(value)

    if finds:
        return finds[-1]
    else:
        return None


def query_flop(task_name):
    """
    Query number of float operation of a task.
    use this function to avoid the dependency of autotvm

    Parameters
    ----------
    task_name: string

    Returns
    ------
    flop: int
    """
    res_table = {
        "resnet.C1.B1": 236027904,
        "resnet.C2.B1": 231211008,
        "resnet.C3.B1": 25690112,
        "resnet.C4.B1": 115605504,
        "resnet.C5.B1": 12845056,
        "resnet.C6.B1": 231211008,
        "resnet.C7.B1": 115605504,
        "resnet.C8.B1": 12845056,
        "resnet.C9.B1": 231211008,
        "resnet.C10.B1": 115605504,
        "resnet.C11.B1": 12845056,
        "resnet.C12.B1": 231211008,

        'mobilenet.D1.B1': 7225344,
        'mobilenet.D2.B1': 3612672,
        'mobilenet.D3.B1': 7225344,
        'mobilenet.D4.B1': 1806336,
        'mobilenet.D5.B1': 3612672,
        'mobilenet.D6.B1': 903168,
        'mobilenet.D7.B1': 1806336,
        'mobilenet.D8.B1': 451584,
        'mobilenet.D9.B1': 903168,

        "other.DEN1": 1024 * 1024 * 1024 * 2,
    }

    if task_name.count('.') == 3:
        task_name = task_name[:task_name.rindex('.')]

    return res_table[task_name]


def create_parent_directory(filename):
    """Create missing parent directories for a filename"""
    parent = os.path.split(filename)[0]

    if not os.path.isdir(parent):
        # make directory
        splits = os.path.split(parent)
        for j in range(1, len(splits)+1):
            path = os.path.join(*splits[:j])
            if not os.path.isdir(path):
                os.mkdir(path)


def enhance_color(color, h=1, l=1, s=1):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    h, l, s = h * c[0], l * c[1], s * c[2]
    h, l, s = [max(min(x, 1), 0) for x in [h, l, s]]

    return colorsys.hls_to_rgb(h, l, s)



def log_to_dashboard(meta_job_uuid, meta_job_name, job_name, tuning_options, device,
                     host='fleet', port=6379):
    """Log the tuning records to a dashboard JSON.

    Parameters
    ----------
    meta_job_uuid: str
        The job uuid for the meta task (e.g., a whole network)
    meta_job_name: str
        e.g., 'resnet18'
    job_name: str
        e.g., 'resnet18.C1.B1'
    tuning_options: dict
        Options passed to the tuner, e.g., tuner type, population size (for GA), etc.
    device: str
        e.g. 'titanx', '1080ti'
    """
    import redis

    whole_db = redis.StrictRedis(host=host, port=port, db=1)
    meta_info_db = redis.StrictRedis(host=host, port=port, db=2)

    class DbEntry:
        def __init__(self):
            self.meta_json = dict()
            self.result_json = dict()
            self.is_first = True

        def dump_to_db(self):
            meta_info_db.set(self.meta_json['job_uuid'], json.dumps(self.meta_json))
            whole_db.set(self.meta_json['job_uuid'], json.dumps(dict(self.meta_json, **self.result_json)))

        def __del__(self):
            self.meta_json['ttl'] = 0
            self.dump_to_db()

    entry = DbEntry()
    entry.meta_json['meta_job_uuid'] = meta_job_uuid
    entry.meta_json['meta_job_name'] = meta_job_name
    entry.meta_json['job_uuid'] = str(uuid.uuid4())
    entry.meta_json['job_name'] = job_name
    entry.meta_json['device'] = device

    entry.meta_json['tuning_options'] = {
        'tuner': tuning_options['tuner'],
        'early_stopping': tuning_options['early_stopping'],
        'n_trial': tuning_options['n_trial'],
    }

    def _callback(tuner, inputs, results):
        if entry.is_first:
            entry.is_first = False
            entry.meta_json['target'] = str(inputs[0].target)
            config = inputs[0][2].to_json_dict()
            entry.meta_json['code_hash'] = config['c']
            entry.meta_json['template_strategy'] = config['t']
            entry.meta_json['workload'] = inputs[0].task.workload
            entry.meta_json['workload_op_count'] = inputs[0].task.flop
            entry.meta_json['peak'] = 0.0
            entry.meta_json['ttl'] = None
            entry.meta_json['start_time'] = min([x.timestamp for x in results])

            entry.result_json['results'] = list()
            entry.result_json['max_results'] = list()

        for res in results:
            res_list = list()
            if res.error_no == 0:
                res_mean = np.mean(res.costs)
                res_list.append(res_mean)
                res_list.append(inputs[0].task.flop/res_mean)

                perf = entry.meta_json['workload_op_count'] / res_mean
                if perf > entry.meta_json['peak']:
                    entry.meta_json['peak'] = perf
            else:
                res_list.append(-1)
                res_list.append(0.0)

            res_list.append(res.error_no)
            res_list.append(res.all_cost)
            res_list.append(res.timestamp)

            entry.result_json['results'].append(tuple(res_list))
            entry.result_json['max_results'].append(entry.meta_json['peak'])

        entry.meta_json['ttl'] = tuner.ttl
        entry.meta_json['trial'] = len(entry.result_json['results'])
        entry.meta_json['end_time'] = max([x.timestamp for x in results])

        entry.dump_to_db()

    return _callback


trans_table = {
"('conv2d', (1, 3, 224, 224, 'float32'), (32, 3, 3, 3, 'float32'), (2, 2), (1, 1), 'NCHW', 'float32')": "mobilenet.C1",
"('depthwise_conv2d_nchw', (1, 32, 112, 112, 'float32'), (32, 1, 3, 3, 'float32'), (1, 1), (1, 1), 'float32')": "mobilenet.D1",
"('conv2d', (1, 32, 112, 112, 'float32'), (64, 32, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C2",
"('depthwise_conv2d_nchw', (1, 64, 112, 112, 'float32'), (64, 1, 3, 3, 'float32'), (2, 2), (1, 1), 'float32')": "mobilenet.D2",
"('conv2d', (1, 64, 56, 56, 'float32'), (128, 64, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C3",
"('depthwise_conv2d_nchw', (1, 128, 56, 56, 'float32'), (128, 1, 3, 3, 'float32'), (1, 1), (1, 1), 'float32')": "mobilenet.D3",
"('conv2d', (1, 128, 56, 56, 'float32'), (128, 128, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C4",
"('depthwise_conv2d_nchw', (1, 128, 56, 56, 'float32'), (128, 1, 3, 3, 'float32'), (2, 2), (1, 1), 'float32')": "mobilenet.D4",
"('conv2d', (1, 128, 28, 28, 'float32'), (256, 128, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C5",
"('depthwise_conv2d_nchw', (1, 256, 28, 28, 'float32'), (256, 1, 3, 3, 'float32'), (1, 1), (1, 1), 'float32')": "mobilenet.D5",
"('conv2d', (1, 256, 28, 28, 'float32'), (256, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C6",
"('depthwise_conv2d_nchw', (1, 256, 28, 28, 'float32'), (256, 1, 3, 3, 'float32'), (2, 2), (1, 1), 'float32')": "mobilenet.D6",
"('conv2d', (1, 256, 14, 14, 'float32'), (512, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C7",
"('depthwise_conv2d_nchw', (1, 512, 14, 14, 'float32'), (512, 1, 3, 3, 'float32'), (1, 1), (1, 1), 'float32')": "mobilenet.D7",
"('conv2d', (1, 512, 14, 14, 'float32'), (512, 512, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C8",
"('depthwise_conv2d_nchw', (1, 512, 14, 14, 'float32'), (512, 1, 3, 3, 'float32'), (2, 2), (1, 1), 'float32')": "mobilenet.D8",
"('conv2d', (1, 512, 7, 7, 'float32'), (1024, 512, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C9",
"('depthwise_conv2d_nchw', (1, 1024, 7, 7, 'float32'), (1024, 1, 3, 3, 'float32'), (1, 1), (1, 1), 'float32')": "mobilenet.D9",
"('conv2d', (1, 1024, 7, 7, 'float32'), (1024, 1024, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "mobilenet.C10",
"('conv2d', (1, 3, 224, 224, 'float32'), (64, 3, 3, 3, 'float32'), (2, 2), (1, 1), 'NCHW', 'float32')": "squeezenet_v1.1.C1",
"('conv2d', (1, 64, 55, 55, 'float32'), (16, 64, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C2",
"('conv2d', (1, 16, 55, 55, 'float32'), (64, 16, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C3",
"('conv2d', (1, 16, 55, 55, 'float32'), (64, 16, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "squeezenet_v1.1.C4",
"('conv2d', (1, 128, 55, 55, 'float32'), (16, 128, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C5",
"('conv2d', (1, 128, 27, 27, 'float32'), (32, 128, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C6",
"('conv2d', (1, 32, 27, 27, 'float32'), (128, 32, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C7",
"('conv2d', (1, 32, 27, 27, 'float32'), (128, 32, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "squeezenet_v1.1.C8",
"('conv2d', (1, 256, 27, 27, 'float32'), (32, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C9",
"('conv2d', (1, 256, 13, 13, 'float32'), (48, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C10",
"('conv2d', (1, 48, 13, 13, 'float32'), (192, 48, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C11",
"('conv2d', (1, 48, 13, 13, 'float32'), (192, 48, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "squeezenet_v1.1.C12",
"('conv2d', (1, 384, 13, 13, 'float32'), (48, 384, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C13",
"('conv2d', (1, 384, 13, 13, 'float32'), (64, 384, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C14",
"('conv2d', (1, 64, 13, 13, 'float32'), (256, 64, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C15",
"('conv2d', (1, 64, 13, 13, 'float32'), (256, 64, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "squeezenet_v1.1.C16",
"('conv2d', (1, 512, 13, 13, 'float32'), (64, 512, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C17",
"('conv2d', (1, 512, 13, 13, 'float32'), (1000, 512, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "squeezenet_v1.1.C18",
"('conv2d', (1, 3, 224, 224, 'float32'), (64, 3, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C1",
"('conv2d', (1, 64, 224, 224, 'float32'), (64, 64, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C2",
"('conv2d', (1, 64, 112, 112, 'float32'), (128, 64, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C3",
"('conv2d', (1, 128, 112, 112, 'float32'), (128, 128, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C4",
"('conv2d', (1, 128, 56, 56, 'float32'), (256, 128, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C5",
"('conv2d', (1, 256, 56, 56, 'float32'), (256, 256, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C6",
"('conv2d', (1, 256, 28, 28, 'float32'), (512, 256, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C7",
"('conv2d', (1, 512, 28, 28, 'float32'), (512, 512, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C8",
"('conv2d', (1, 512, 14, 14, 'float32'), (512, 512, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "vgg-16.C9",
"('conv2d', (1, 3, 299, 299, 'float32'), (32, 3, 3, 3, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "inception_v3.C1",
"('conv2d', (1, 32, 149, 149, 'float32'), (32, 32, 3, 3, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C2",
"('conv2d', (1, 32, 147, 147, 'float32'), (64, 32, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "inception_v3.C3",
"('conv2d', (1, 64, 73, 73, 'float32'), (80, 64, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C4",
"('conv2d', (1, 80, 73, 73, 'float32'), (192, 80, 3, 3, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C5",
"('conv2d', (1, 192, 35, 35, 'float32'), (64, 192, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C6",
"('conv2d', (1, 192, 35, 35, 'float32'), (48, 192, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C7",
"('conv2d', (1, 48, 35, 35, 'float32'), (64, 48, 5, 5, 'float32'), (1, 1), (2, 2), 'NCHW', 'float32')": "inception_v3.C8",
"('conv2d', (1, 64, 35, 35, 'float32'), (96, 64, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "inception_v3.C9",
"('conv2d', (1, 96, 35, 35, 'float32'), (96, 96, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "inception_v3.C10",
"('conv2d', (1, 192, 35, 35, 'float32'), (32, 192, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C11",
"('conv2d', (1, 256, 35, 35, 'float32'), (64, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C12",
"('conv2d', (1, 256, 35, 35, 'float32'), (48, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C13",
"('conv2d', (1, 288, 35, 35, 'float32'), (64, 288, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C14",
"('conv2d', (1, 288, 35, 35, 'float32'), (48, 288, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C15",
"('conv2d', (1, 288, 35, 35, 'float32'), (384, 288, 3, 3, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "inception_v3.C16",
"('conv2d', (1, 96, 35, 35, 'float32'), (96, 96, 3, 3, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "inception_v3.C17",
"('conv2d', (1, 768, 17, 17, 'float32'), (192, 768, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C18",
"('conv2d', (1, 768, 17, 17, 'float32'), (128, 768, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C19",
"('conv2d', (1, 128, 17, 17, 'float32'), (128, 128, 1, 7, 'float32'), (1, 1), (0, 3), 'NCHW', 'float32')": "inception_v3.C20",
"('conv2d', (1, 128, 17, 17, 'float32'), (192, 128, 7, 1, 'float32'), (1, 1), (3, 0), 'NCHW', 'float32')": "inception_v3.C21",
"('conv2d', (1, 128, 17, 17, 'float32'), (128, 128, 7, 1, 'float32'), (1, 1), (3, 0), 'NCHW', 'float32')": "inception_v3.C22",
"('conv2d', (1, 128, 17, 17, 'float32'), (192, 128, 1, 7, 'float32'), (1, 1), (0, 3), 'NCHW', 'float32')": "inception_v3.C23",
"('conv2d', (1, 768, 17, 17, 'float32'), (160, 768, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C24",
"('conv2d', (1, 160, 17, 17, 'float32'), (160, 160, 1, 7, 'float32'), (1, 1), (0, 3), 'NCHW', 'float32')": "inception_v3.C25",
"('conv2d', (1, 160, 17, 17, 'float32'), (192, 160, 7, 1, 'float32'), (1, 1), (3, 0), 'NCHW', 'float32')": "inception_v3.C26",
"('conv2d', (1, 160, 17, 17, 'float32'), (160, 160, 7, 1, 'float32'), (1, 1), (3, 0), 'NCHW', 'float32')": "inception_v3.C27",
"('conv2d', (1, 160, 17, 17, 'float32'), (192, 160, 1, 7, 'float32'), (1, 1), (0, 3), 'NCHW', 'float32')": "inception_v3.C28",
"('conv2d', (1, 192, 17, 17, 'float32'), (192, 192, 1, 7, 'float32'), (1, 1), (0, 3), 'NCHW', 'float32')": "inception_v3.C29",
"('conv2d', (1, 192, 17, 17, 'float32'), (192, 192, 7, 1, 'float32'), (1, 1), (3, 0), 'NCHW', 'float32')": "inception_v3.C30",
"('conv2d', (1, 192, 17, 17, 'float32'), (320, 192, 3, 3, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "inception_v3.C31",
"('conv2d', (1, 192, 17, 17, 'float32'), (192, 192, 3, 3, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "inception_v3.C32",
"('conv2d', (1, 1280, 8, 8, 'float32'), (320, 1280, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C33",
"('conv2d', (1, 1280, 8, 8, 'float32'), (384, 1280, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C34",
"('conv2d', (1, 384, 8, 8, 'float32'), (384, 384, 1, 3, 'float32'), (1, 1), (0, 1), 'NCHW', 'float32')": "inception_v3.C35",
"('conv2d', (1, 384, 8, 8, 'float32'), (384, 384, 3, 1, 'float32'), (1, 1), (1, 0), 'NCHW', 'float32')": "inception_v3.C36",
"('conv2d', (1, 1280, 8, 8, 'float32'), (448, 1280, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C37",
"('conv2d', (1, 448, 8, 8, 'float32'), (384, 448, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "inception_v3.C38",
"('conv2d', (1, 1280, 8, 8, 'float32'), (192, 1280, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C39",
"('conv2d', (1, 2048, 8, 8, 'float32'), (320, 2048, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C40",
"('conv2d', (1, 2048, 8, 8, 'float32'), (384, 2048, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C41",
"('conv2d', (1, 2048, 8, 8, 'float32'), (448, 2048, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C42",
"('conv2d', (1, 2048, 8, 8, 'float32'), (192, 2048, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "inception_v3.C43",
"('conv2d', (1, 3, 224, 224, 'float32'), (64, 3, 7, 7, 'float32'), (2, 2), (3, 3), 'NCHW', 'float32')": "resnet-50.C1",
"('conv2d', (1, 64, 56, 56, 'float32'), (64, 64, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C2",
"('conv2d', (1, 64, 56, 56, 'float32'), (64, 64, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-50.C3",
"('conv2d', (1, 64, 56, 56, 'float32'), (256, 64, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C4",
"('conv2d', (1, 256, 56, 56, 'float32'), (64, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C5",
"('conv2d', (1, 256, 56, 56, 'float32'), (128, 256, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-50.C6",
"('conv2d', (1, 128, 28, 28, 'float32'), (128, 128, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-50.C7",
"('conv2d', (1, 256, 56, 56, 'float32'), (512, 256, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-50.C8",
"('conv2d', (1, 128, 28, 28, 'float32'), (512, 128, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C9",
"('conv2d', (1, 512, 28, 28, 'float32'), (128, 512, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C10",
"('conv2d', (1, 512, 28, 28, 'float32'), (256, 512, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-50.C11",
"('conv2d', (1, 256, 14, 14, 'float32'), (256, 256, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-50.C12",
"('conv2d', (1, 512, 28, 28, 'float32'), (1024, 512, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-50.C13",
"('conv2d', (1, 256, 14, 14, 'float32'), (1024, 256, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C14",
"('conv2d', (1, 1024, 14, 14, 'float32'), (256, 1024, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C15",
"('conv2d', (1, 1024, 14, 14, 'float32'), (512, 1024, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-50.C16",
"('conv2d', (1, 512, 7, 7, 'float32'), (512, 512, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-50.C17",
"('conv2d', (1, 1024, 14, 14, 'float32'), (2048, 1024, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-50.C18",
"('conv2d', (1, 512, 7, 7, 'float32'), (2048, 512, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C19",
"('conv2d', (1, 2048, 7, 7, 'float32'), (512, 2048, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-50.C20",
"('conv2d', (1, 3, 224, 224, 'float32'), (64, 3, 7, 7, 'float32'), (2, 2), (3, 3), 'NCHW', 'float32')": "resnet-18.C1",
"('conv2d', (1, 64, 56, 56, 'float32'), (64, 64, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-18.C2",
"('conv2d', (1, 64, 56, 56, 'float32'), (64, 64, 1, 1, 'float32'), (1, 1), (0, 0), 'NCHW', 'float32')": "resnet-18.C3",
"('conv2d', (1, 64, 56, 56, 'float32'), (128, 64, 3, 3, 'float32'), (2, 2), (1, 1), 'NCHW', 'float32')": "resnet-18.C4",
"('conv2d', (1, 64, 56, 56, 'float32'), (128, 64, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-18.C5",
"('conv2d', (1, 128, 28, 28, 'float32'), (128, 128, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-18.C6",
"('conv2d', (1, 128, 28, 28, 'float32'), (256, 128, 3, 3, 'float32'), (2, 2), (1, 1), 'NCHW', 'float32')": "resnet-18.C7",
"('conv2d', (1, 128, 28, 28, 'float32'), (256, 128, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-18.C8",
"('conv2d', (1, 256, 14, 14, 'float32'), (256, 256, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-18.C9",
"('conv2d', (1, 256, 14, 14, 'float32'), (512, 256, 3, 3, 'float32'), (2, 2), (1, 1), 'NCHW', 'float32')": "resnet-18.C10",
"('conv2d', (1, 256, 14, 14, 'float32'), (512, 256, 1, 1, 'float32'), (2, 2), (0, 0), 'NCHW', 'float32')": "resnet-18.C11",
"('conv2d', (1, 512, 7, 7, 'float32'), (512, 512, 3, 3, 'float32'), (1, 1), (1, 1), 'NCHW', 'float32')": "resnet-18.C12",
}

def workload_to_name(wkl):
    """translate workload tuple to readable name"""

    if isinstance(wkl, str):
        return trans_table.get(wkl, str(wkl))

    wkl = list(wkl)
    batch_size = wkl[1][0]
    wkl[1] = list(wkl[1])
    wkl[1][0] = 1
    match_str = str(wkl).replace('[', '(').replace(']', ')')

    name = workload_to_name(match_str)
    return trans_table.get(match_str, str(wkl)) + ".B%d" % batch_size

