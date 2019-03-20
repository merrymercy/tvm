"""Test the effect of auto-scheduler (parallel, vectorization, fusion, tiling)"""
import logging
import sys
import re

import numpy as np

import topi
import tvm
from tvm import autotvm
from tvm.autotvm.auto_schedule.common import AutoScheduleOptions as opts
from topi.util import get_const_tuple


def _test_speed(bufs, target):
    with target:
        s = autotvm.create_schedule(bufs)
        #s = tvm.create_schedule([bufs[-1].op])
        func = tvm.build(s, bufs)

    ctx = tvm.context(str(target))
    args = []
    for x in bufs:
        args.append(tvm.nd.array(np.random.randn(*get_const_tuple(x.shape)).astype(x.dtype),
                                 ctx=ctx))

    timer = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=1000)
    return timer(*args).mean


def test_parallel_vec():
    """test parallel and vectorize"""
    A = tvm.placeholder((16, 16), name='A')
    B = tvm.compute((16, 16), lambda i, j: A[i][j] + 1, name='B')

    ## CPU
    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, B])
        stmt = tvm.lower(s, [A, B], simple_mode=True)

    assert isinstance(stmt.body.body.body.index, tvm.expr.Ramp)  # vec
    assert stmt.body.for_type == 1                               # parallel
    assert "if " not in str(stmt)

    ## GPU
    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, B])
        stmt = tvm.lower(s, [A, B], simple_mode=True)

    assert stmt.body.node.var.name == "blockIdx.x"
    assert stmt.body.attr_key == "thread_extent"
    assert stmt.body.body.node.var.name == "threadIdx.x"
    assert stmt.body.body.attr_key == "thread_extent"
    assert "if " not in str(stmt)


def test_inline():
    """test inline"""
    A = tvm.placeholder((16,), name='A')
    B = tvm.compute((16,), lambda i: A[i], name='B_Buffer')
    C = tvm.compute((16,), lambda j: B[j], name='C')

    ## CPU
    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, C])
        stmt = tvm.lower(s, [A, C], simple_mode=True)

    assert 'B_Buffer' not in str(stmt)

    ## GPU
    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, C])
        stmt = tvm.lower(s, [A, C], simple_mode=True)

    assert 'B_Buffer' not in str(stmt)


def test_tune_simple_compute():
    """test tuning compute_at"""

    @autotvm.template
    def blur():
        A = tvm.placeholder((18, 18), name='A')
        B = tvm.compute((18, 16), lambda i, j: (A[i][j] + A[i][j+1] + A[i][j+2])/3, name='B')
        C = tvm.compute((16, 16), lambda i, j: (B[i][j] + B[i+1][j] + B[i+2][j])/3, name='C')

        with autotvm.AutoScheduleOptions(tuning_level=3):
            s = autotvm.create_schedule([A, C])

        return s, [A, C]

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    # CPU
    flags = [False, False]

    def check_call_back(_, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no != 0:
                continue

            with inp.target:
                s, args = inp.task.instantiate(inp.config)
            stmt = tvm.lower(s, args, simple_mode=True)
            if isinstance(stmt, tvm.stmt.AttrStmt):
                flags[0] = True
            elif isinstance(stmt, tvm.stmt.ProducerConsumer):
                flags[1] = True

    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=1))

    tsk = autotvm.task.create(blur, args=(), target='llvm')
    tuner = autotvm.tuner.RandomTuner(tsk)
    tuner.tune(n_trial=8,
               measure_option=measure_option,
               callbacks=[check_call_back])
    assert flags[0] and flags[1]

    # GPU
    if tvm.gpu(0).exist:
        flags = [False, False]
        measure_option = autotvm.measure_option(
            builder='local',
            runner=autotvm.LocalRunner(number=1))

        tsk = autotvm.task.create(blur, args=(), target='cuda')
        tuner = autotvm.tuner.RandomTuner(tsk)
        tuner.tune(n_trial=2,
                   measure_option=measure_option,
                   callbacks=[check_call_back])
        assert flags[0] and flags[1]


def test_norm():
    """test norm"""
    D1 = D2 = D3 = 128

    A = tvm.placeholder((D1, D2, D3), name='A')

    r1 = tvm.reduce_axis((0, D1), name='r1')
    r2 = tvm.reduce_axis((0, D2), name='r2')
    r3 = tvm.reduce_axis((0, D3), name='r3')

    B = tvm.compute((1,), lambda i: tvm.sum(A[r1, r2, r3] * A[r1, r2, r3],
                                            axis=[r1, r2, r3]), name='C')
    B = topi.sqrt(B)

    ## CPU
    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, B])
        stmt = tvm.lower(s, [A, B], simple_mode=True)

    stmt = str(stmt)
    assert "parallel" in stmt
    assert "ramp" in stmt
    assert len(re.findall("\nproduce", stmt)) == 1  # test fuse

    ## GPU
    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, B])
        stmt = tvm.lower(s, [A, B], simple_mode=True)

    stmt = str(stmt)
    assert len(re.findall("\nproduce", stmt)) == 0  # test fuse


def test_more_reduction():
    """test more reduction ops"""
    def pool():
        A = tvm.placeholder((1, 256, 7, 7), name='A')
        D = topi.nn.pool(A, (2, 2), (1, 1), (0, 0, 0, 0), 'max')
        D = topi.nn.relu(D)
        return A, D

    def global_pool():
        A = tvm.placeholder((1, 256, 7, 7), name='A')
        D = topi.nn.global_pool(A, 'max')
        D = topi.nn.relu(D)
        return A, D

    def softmax():
        A = tvm.placeholder((1, 20, 1000), name='A')
        D = topi.nn.softmax(A, axis=1)
        D = topi.nn.relu(D)
        return A, D

    def argmin():
        A = tvm.placeholder((1, 128, 128), name='A')
        D = topi.argmin(A, axis=(1, 2))
        D = topi.nn.relu(D)
        return A, D

    def ele_max():
        A = tvm.placeholder((128, 128), name='A')
        D = topi.max(A)
        return A, D

    test_cast_list = [
        pool(),
        global_pool(),
        softmax(),
        argmin(),
        ele_max()
    ]

    ## CPU
    for buffers in test_cast_list:
        with tvm.target.create('llvm'):
            s = autotvm.create_schedule(buffers)
            stmt = tvm.lower(s, buffers, simple_mode=True)
            stmt = str(stmt)
            assert "parallel" in stmt or "ramp" in stmt
            assert "if " not in stmt, stmt

    ## GPU
    with tvm.target.create('cuda'):
        for buffers in [pool()]:
            s = autotvm.create_schedule(buffers)
            stmt = tvm.lower(s, buffers, simple_mode=True)
            stmt = str(stmt)
            assert "allreduce" not in stmt

        for buffers in [global_pool(), softmax()]:
            s = autotvm.create_schedule(buffers)
            stmt = tvm.lower(s, buffers, simple_mode=True)
            stmt = str(stmt)
            assert "allreduce" not in stmt
            assert "if " not in stmt

        for buffers in [argmin()]:
            s = autotvm.create_schedule(buffers)
            stmt = tvm.lower(s, buffers, simple_mode=True)
            stmt = str(stmt)
            assert "allreduce" in stmt


def test_matmul():
    """test matmul"""
    A = tvm.placeholder((128, 128), name='A')
    B = tvm.placeholder((128, 128), name='B')
    k = tvm.reduce_axis((0, 128), 'k')
    C = tvm.compute((128, 128), lambda i, j: tvm.sum(A[i, k] * B[k, j], k), name='C')

    ## CPU
    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, B, C])
        stmt = tvm.lower(s, [A, B, C], simple_mode=True)

    stmt = str(stmt)
    assert "parallel" in stmt
    assert "ramp" in stmt

    ## GPU
    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, B, C])
        stmt = tvm.lower(s, [A, B, C], simple_mode=True)

    stmt = str(stmt)
    assert "if " not in stmt
    assert "shared" in stmt


def test_dense_fuse():
    """test dense. This is matmul in another layout"""
    A = tvm.placeholder((4, 4096), name='A')
    W = tvm.placeholder((1000, 4096), name='W')

    O = topi.nn.dense(A, W)
    O = topi.nn.relu(O)

    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, W, O])
        stmt = tvm.lower(s, [A, W, O], simple_mode=True)

    stmt = str(stmt)
    assert "parallel" in stmt
    assert "ramp" in stmt

    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, W, O])
        stmt = tvm.lower(s, [A, W, O], simple_mode=True)

    stmt = str(stmt)
    assert "shared" in stmt

def test_conv_nchw():
    """Test conv with padding"""
    A = tvm.placeholder((4, 64, 56, 56), name='A')
    W = tvm.placeholder((128, 64, 3, 3), name='W')
    bias = tvm.placeholder((128,), name='bias')

    O = topi.nn.conv2d(A, W, (1, 1), (1, 1),
                       (1, 1), layout='NCHW', out_dtype='float32')
    O = topi.add(O, topi.expand_dims(bias, axis=1, num_newaxis=2))

    ## CPU
    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, W, bias, O])
        stmt = tvm.lower(s, [A, W, bias, O], simple_mode=True)

    stmt = str(stmt)
    assert len(re.findall("parallel", stmt)) == 2   # test parallel
    assert "ramp" in stmt                           # test vectorization
    assert len(re.findall("\nproduce", stmt)) == 2  # test fuse

    ## GPU
    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, W, bias, O])
        stmt = tvm.lower(s, [A, W, bias, O], simple_mode=True)

    shared_mem = 0
    for item in re.findall("allocate .*\.shared\[float32 \* (\d*)\]", str(stmt)):
        shared_mem += int(item) * 4
    assert shared_mem <= opts.MAX_SHARED_MEMORY


def test_conv3d():
    """3D convolusion"""
    N, CI, D, H, W, CO, kernel_size, stride, padding, dtype =\
        2, 32, 28, 28, 28, 64, 3, 1, 1, 'float32'

    KD = KH = KW = kernel_size
    SD = SH = SW = stride
    PD = PH = PW = padding

    A = tvm.placeholder((N, CI, D, H, W), dtype=dtype, name='A')
    B = tvm.placeholder((CO, CI, KD, KH, KW), dtype=dtype, name='B')
    bias = tvm.placeholder((CO,), name='bias')

    ci = tvm.reduce_axis((0, CI), 'rc')
    kd = tvm.reduce_axis((0, KD), 'kd')
    kh = tvm.reduce_axis((0, KH), 'kh')
    kw = tvm.reduce_axis((0, KW), 'kw')

    OD = (D + 2 * PD - KD) // SD + 1
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    C = tvm.compute((N, CO, OD, OH, OW), lambda n, co, od, oh, ow:
            tvm.sum(A[n, ci, od * SD + kd, oh * SH + kh, ow * SW + kw] *
                    B[co, ci, kd, kh, kw], axis=[ci, kd, kh, kw]), name='C')
    C = topi.add(C, topi.expand_dims(bias, axis=1, num_newaxis=3))

    ## CPU
    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, B, bias, C])
        stmt = tvm.lower(s, [A, B, bias, C], simple_mode=True)

    stmt = str(stmt)
    assert "parallel" in stmt                       # test parallel
    assert "ramp" in stmt                           # test vectorization
    assert len(re.findall("^produce", stmt)) == 1   # test fuse

    ## GPU
    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, B, bias, C])
        stmt = tvm.lower(s, [A, B, bias, C], simple_mode=True)

    shared_mem = 0
    for item in re.findall("allocate .*\.shared\[float32 \* (\d*)\]", str(stmt)):
        shared_mem += int(item) * 4
    assert shared_mem <= opts.MAX_SHARED_MEMORY


def test_depthwise_conv2d():
    A = tvm.placeholder((4, 64, 56, 56), name='A')
    W = tvm.placeholder((64, 1, 3, 3), name='W')
    bias = tvm.placeholder((64,), name='bias')

    O = topi.nn.depthwise_conv2d_nchw(A, W, (1, 1), (1, 1),
                                      (1, 1), out_dtype='float32')
    O = topi.add(O, topi.expand_dims(bias, axis=1, num_newaxis=2))

    ## CPU
    with tvm.target.create('llvm'):
        s = autotvm.create_schedule([A, W, bias, O])
        stmt = tvm.lower(s, [A, W, bias, O], simple_mode=True)

    stmt = str(stmt)
    assert len(re.findall("parallel", stmt)) == 2   # test parallel
    assert len(re.findall("\nproduce", stmt)) == 2  # test fuse

    ## GPU
    with tvm.target.create('cuda'):
        s = autotvm.create_schedule([A, W, bias, O])
        stmt = tvm.lower(s, [A, W, bias, O], simple_mode=True)

    stmt = str(stmt)
    assert "shared" in stmt

def test_pack():
    """Test auto-packing"""
    pass

if __name__ == "__main__":
    test_parallel_vec()
    test_inline()
    test_tune_simple_compute()
    test_norm()
    test_more_reduction()
    test_matmul()
    test_dense_fuse()
    test_conv_nchw()
    test_conv3d()
    test_depthwise_conv2d()

    test_pack()
