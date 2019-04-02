"""Schedule template for GPU"""

import numpy as np

from .... import target as _target, tensor as _tensor, expr as _expr, \
    ir_pass as _ir_pass, arith as _arith, thread_axis
from ...task.space import get_factors, ConfigSpace, FallbackConfigEntity
from ..common import AutoScheduleOptions as opts, get_axis_length, tuning_level
from .generic import schedule_simple_reduce, schedule_complex_reduce, \
    schedule_other_root, schedule_tune_direct_compute

def _parallel_spatial(s, op, axes, num_axes=None, axis_name="threadIdx.x"):
    """Parallel and bind spatial axes"""
    num_axes = num_axes or opts.MAX_GPU_THREADS
    prod = np.prod([get_axis_length(x) for x in s[op].op.axis])

    # set upper bound of the number of axes
    while num_axes > 16 and num_axes * 4 > prod:
        num_axes //= 2

    # consider divisibility
    num_axes = [x for x in get_factors(prod) if x <= num_axes][-1]

    fuse_all = s[op].fuse(*axes)
    bx, tx = s[op].split(fuse_all, num_axes)

    s[op].bind(bx, thread_axis("blockIdx.x"))
    s[op].bind(tx, thread_axis(axis_name))

    return tx

def _get_byte_size(dtype):
    """Get the number of bytes of a dtype"""
    table = {
        "float64": 8, "float32": 4, "float16": 2,
        "int64": 8, "int32": 4, "int16": 2, "int8": 1,
    }
    return table.get(dtype, 8)

def _estimate_memory(op, axis_lens):
    """Estimate touched memory of an ComputeOp"""
    assert isinstance(op, _tensor.ComputeOp), "Only support compute op"

    analyzer = _arith.Analyzer()
    for i, axis in enumerate(op.axis):
        analyzer.update(axis.var, _arith.ConstIntBound(0, axis_lens[i]))
    base = len(op.axis)
    for i, axis in enumerate(op.reduce_axis):
        analyzer.update(axis.var, _arith.ConstIntBound(0, axis_lens[i + base]))

    read_list = []

    def _check(stmt):
        if isinstance(stmt, _expr.Call) and stmt.call_type == _expr.Call.Halide:
            tmp = 1
            for x in stmt.args:
                bound = analyzer.const_int_bound(x)
                tmp *= bound.max_value - bound.min_value
            read_list.append(tmp * _get_byte_size(stmt.dtype))

    _ir_pass.PostOrderVisit(op.body[0].source[0], _check)
    return sum(read_list)


@schedule_other_root.register('gpu')
def schedule_other_root_gpu(s, cfg, node):
    """Schedule other compute type (e.g. elemwise, broadcast)

    Parameters
    ----------
    s: Schedule
        The schedule
    cfg: ConfigSpace
        The current configuration for autotvm template
    node: StageNode
        The node(op) to schedule
    """
    op = node.op

    # todo(lmzheng): consider reordering?
    _parallel_spatial(s, op, s[op].op.axis)


@schedule_tune_direct_compute.register('gpu')
def schedule_tune_direct_compute(s, cfg, node):
    """Schedule tunable direct compute op (e.g. elemwise with tunable location)

    Parameters
    ----------
    s: Schedule
        The schedule
    cfg: ConfigSpace
        The current configuration for autotvm template
    node: StageNode
        The node(op) to schedule
    """
    op = node.op

    # ==================== Define Configuration Space ====================
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        cfg.define_knob("#2-" + node.name + '_compute_loc', [0, 1])

    # ========================== Heuristic Fill ==========================
    if tuning_level(cfg) < 2:
        cfg["#2-" + node.name + '_compute_loc'].val = 0

    # =========================== Apply Config ===========================
    val = cfg["#2-" + node.name + '_compute_loc'].val
    if val == 0:
        s[op].compute_inline()
    elif val == 1:
        _parallel_spatial(s, op, s[op].op.axis)


@schedule_simple_reduce.register('gpu')
def schedule_simple_reduce_gpu(s, cfg, node):
    """Schedule simple reduction op (e.g. softmax, min, max)

    Parameters
    ----------
    s: Schedule
        The schedule
    cfg: ConfigSpace
        The current configuration for autotvm template
    node: StageNode
        The node(op) to schedule
    """
    op = node.op
    output = node.compute_at_loc.op

    # ==================== Define Configuration Space ====================
    prefix = "#2-" + node.name
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        # parallel on spatial axis or reduction axis
        cfg.define_knob(prefix + '_parallel_loc', ['spatial', 'reduction'])
        cfg.define_knob(prefix + '_parallel_num', [32, 64, 128, 256, 512])

    # ========================== Heuristic Fill ==========================
    if isinstance(cfg, FallbackConfigEntity) or opts.TUNING_LEVEL <= 2:
        spatial_prod = np.prod([get_axis_length(x) for x in s[op].op.axis])
        reduction_prod = np.prod([get_axis_length(x) for x in s[op].op.reduce_axis])

        # decide whether to parallel reduction axes or spatial axes
        if spatial_prod >= get_axis_length(s[op].op.reduce_axis[0]) \
                or reduction_prod < opts.NUM_THREADS * 4:
            cfg[prefix + '_parallel_loc'].val = 'spatial'
        else:
            cfg[prefix + '_parallel_loc'].val = 'reduction'

        if spatial_prod == 1:
            cfg[prefix + "_parallel_num"].val = _target.current_target().max_num_threads
        else:
            cfg[prefix + "_parallel_num"].val = 32

    # =========================== Apply Config ===========================
    def _parallel_reduction(T, output, num_tx):
        fused_reduce = s[T].fuse(*s[T].op.reduce_axis)
        ko, ki = s[T].split(fused_reduce, factor=num_tx)
        TF = s.rfactor(T, ki)
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        tx = s[T].op.reduce_axis[0]
        thread_x = thread_axis('threadIdx.x')
        s[T].bind(tx, thread_x)
        s[TF].compute_at(s[T], tx)

        s[output].set_store_predicate(thread_x.equal(0))
        num_tx = _target.current_target().max_num_threads // cfg[prefix + '_parallel_num'].val
        return _parallel_spatial(s, output, s[output].op.axis, num_tx, "threadIdx.y")

    tensor = op.output(0)

    if cfg[prefix + '_parallel_loc'].val == 'spatial':
        last = _parallel_spatial(s, output, s[output].op.axis)
    else:
        last = _parallel_reduction(tensor, output, cfg[prefix + '_parallel_num'].val)

    if op != output and last is not None:
        s[op].compute_at(s[output], last)

@schedule_complex_reduce.register('gpu')
def schedule_complex_reduce_gpu(s, cfg, node):
    """Schedule simple reduction op (e.g. softmax, min, max)

    Parameters
    ----------
    s: Schedule
        The schedule
    cfg: ConfigSpace
        The current configuration for autotvm template
    node: StageNode
        The node(op) to schedule
    """
    op = node.op
    output = node.compute_at_loc.op.output(0)

    n_sp = len(s[op].op.axis)
    n_rd = len(s[op].op.reduce_axis)
    lens = [get_axis_length(x) for x in s[op].op.axis] + \
           [get_axis_length(x) for x in s[op].op.reduce_axis]
    n_axis = n_sp + n_rd

    # ==================== Define Configuration Space ====================
    prefix = "#2-" + node.name
    tile_level = 4
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        for i, axis in enumerate(s[op].op.axis):
            cfg.define_split(prefix + "_sp_tile_%d" % i, get_axis_length(axis),
                             num_outputs=tile_level)

        for i, axis in enumerate(s[op].op.reduce_axis):
            cfg.define_split(prefix + "_rd_tile_%d" % i, get_axis_length(axis),
                             num_outputs=tile_level-2)

        cfg.define_knob(prefix + 'unroll_explicit', [0, 1])
        cfg.define_knob(prefix + 'auto_unroll_max_step', [0, 512, 1500])

    # ========================== Heuristic Fill ==========================
    if tuning_level(cfg) < 1:
        # todo(lmzheng): The following logic might be wrong?
        # refactor this part by doing more performance experiments

        # 1. Determine block size: use as much shared memory as possible
        # Assume tiling on every dimension can increase the locality or benefit load,
        # we greedily increase tile size for each dimension, in round robin order.
        block_size = [1] * n_axis
        factors = [get_factors(x) for x in lens]

        ct = n_axis
        find = True
        while find:
            find = False
            for i in range(n_axis):
                ct_ = (ct + i) % n_axis
                step = (ct + i) // n_axis

                if len(factors[ct_]) <= step:
                    continue

                tmp = block_size[ct_]
                block_size[ct_] = factors[ct_][step]
                next_mem = _estimate_memory(op, block_size)

                if next_mem > opts.MAX_SHARED_MEMORY:
                    block_size[ct_] = tmp
                    continue

                ct = ct + i + 1
                find = True
                break

        # 2. Allocate thread binding
        # Inside a block, utilize as many threads as possible
        thread_size = [1] * n_sp
        factors = [get_factors(block_size[i]) for i in range(n_sp)]

        ct = n_sp
        find = True
        while find:
            find = False
            for i in range(n_sp):
                ct_ = (ct + i) % n_sp
                step = (ct + i) // n_sp

                if len(factors[ct_]) <= step:
                    continue

                tmp = thread_size[ct_]
                thread_size[ct_] = factors[ct_][step]

                if np.prod(thread_size) > opts.MAX_GPU_THREADS:
                    thread_size[ct_] = tmp
                    continue

                ct = ct + i + 1
                find = True
                break

        # 3. Fill the values in cfg
        for i in range(n_sp):
            cfg[prefix + '_sp_tile_%d' % i].size = [lens[i] // block_size[i], 1, thread_size[i],
                                                    block_size[i] / thread_size[i]]

        for i in range(n_rd):
            cfg[prefix + '_rd_tile_%d' % i].size = [lens[i + n_sp] // block_size[i + n_sp],
                                                    block_size[i + n_sp]]

            # Todo(lmzheng): consider bank conflict?

    # =========================== Apply Config ===========================
    input_tensor = s[op].op.input_tensors
    shared_cache = []

    if output == s[op].op.output(0):
        local = s.cache_write(output, "local")
    else:
        local = s[op].op.output(0)
        s[local].set_scope('local')

    # add cache read and write stage
    for buf in input_tensor:
        shared_cache.append(s.cache_read(buf, "shared", [local]))

    # tile outer spatial axes
    sp_chains = []
    for i in range(n_sp):
        ch = cfg[prefix + '_sp_tile_%d' % i].apply(s, output, s[output].op.axis[i])
        sp_chains.append(ch)

    sp_axes = list(sum(zip(*sp_chains), ()))
    kernel_scope, sp_axes[0] = s[output].split(sp_axes[0], nparts=1)
    s[output].reorder(*sp_axes)

    # tile inner reduction axes
    rd_chains = []
    for i in range(n_rd):
        ch = cfg[prefix + '_rd_tile_%d' % i].apply(s, local, s[local].op.reduce_axis[i])
        rd_chains.append(ch)

    local_sp_axes = list(s[local].op.axis)
    rd_axes = list(sum(zip(*rd_chains), ()))
    s[local].reorder(*(rd_axes + local_sp_axes))

    # bind threads and cooperative fetching
    if n_sp >= 3:
        bz = s[output].fuse(*sp_axes[:n_sp-2])
        by = sp_axes[n_sp - 2]
        bx = sp_axes[n_sp - 1]
        tz = s[output].fuse(*sp_axes[n_sp * 2:n_sp * 3 - 2])
        ty = sp_axes[n_sp * 3 - 2]
        tx = sp_axes[n_sp * 3 - 1]
        s[output].bind(bz, thread_axis('blockIdx.z'))
        s[output].bind(by, thread_axis('blockIdx.y'))
        s[output].bind(bx, thread_axis('blockIdx.x'))
        s[output].bind(tz, thread_axis('threadIdx.z'))
        s[output].bind(ty, thread_axis('threadIdx.y'))
        s[output].bind(tx, thread_axis('threadIdx.x'))

        # cooperative fetching  (todo: smarter binding or better _arith simplification)
        for load in shared_cache:
            fused = s[load].fuse(*s[load].op.axis)
            len_z = round(np.prod([cfg[prefix + '_sp_tile_%d' % i].size[2]
                                   for i in range(0, n_sp-2)]))
            rtz, fused = s[load].split(fused, nparts=len_z)
            rty, fused = s[load].split(fused, nparts=cfg[prefix + '_sp_tile_%d' % (n_sp-2)].size[2])
            rtx, fused = s[load].split(fused, nparts=cfg[prefix + '_sp_tile_%d' % (n_sp-1)].size[2])
            s[load].bind(rtz, thread_axis("threadIdx.z"))
            s[load].bind(rty, thread_axis("threadIdx.y"))
            s[load].bind(rtx, thread_axis("threadIdx.x"))
    elif n_sp == 2:
        by = sp_axes[0]
        bx = sp_axes[1]
        ty = sp_axes[4]
        tx = sp_axes[5]
        s[output].bind(by, thread_axis('blockIdx.y'))
        s[output].bind(bx, thread_axis('blockIdx.x'))
        s[output].bind(ty, thread_axis('threadIdx.y'))
        s[output].bind(tx, thread_axis('threadIdx.x'))

        # cooperative fetching
        for load in shared_cache:
            fused = s[load].fuse(*s[load].op.axis)
            rty, fused = s[load].split(fused, nparts=cfg[prefix + '_sp_tile_0'].size[2])
            rtx, fused = s[load].split(fused, nparts=cfg[prefix + '_sp_tile_1'].size[2])
            s[load].bind(rty, thread_axis("threadIdx.y"))
            s[load].bind(rtx, thread_axis("threadIdx.x"))
    elif n_sp == 1:
        bx = sp_axes[0]
        tx = sp_axes[2]
        s[output].bind(bx, thread_axis('blockIdx.x'))
        s[output].bind(tx, thread_axis('threadIdx.x'))

        # cooperative fetching
        for load in shared_cache:
            fused = s[load].fuse(*s[load].op.axis)
            rtx, fused = s[load].split(fused, nparts=cfg[prefix + '_sp_tile_0'].size[2])
            s[load].bind(rtx, thread_axis("threadIdx.x"))
    else:
        raise RuntimeError("Wrong Operator Definition")

    s[local].compute_at(s[output], tx)
    for load in shared_cache:
        s[load].compute_at(s[local], rd_chains[-1][0])

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg[prefix + 'auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg[prefix + 'unroll_explicit'].val)
