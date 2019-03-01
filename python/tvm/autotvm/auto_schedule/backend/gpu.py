import numpy as np

from .... import target as _target, expr as _expr, tensor as _tensor, thread_axis, ir_pass, schedule
from ...task.space import get_factors, ConfigEntity, ConfigSpace, FallbackConfigEntity
from ..common import AutoScheduleOptions as opts, _get_axis_length
from .generic import schedule_simple_reduce, schedule_complex_reduce, schedule_other_root

def _parallel_spatial(s, op, axes, num_axes=None, axis_name="threadIdx.x"):
    """parallel and bind spatial axes"""
    num_axes = num_axes or opts.MAX_GPU_THREADS
    prod = np.prod([_get_axis_length(x) for x in s[op].op.axis])

    # set range
    while num_axes > 16 and num_axes * 4 > prod:
        num_axes //= 2

    # consider divisibility
    num_axes = [x for x in get_factors(prod) if x <= num_axes][-1]

    fuse_all = s[op].fuse(*axes)
    bx, tx = s[op].split(fuse_all, num_axes)

    s[op].bind(bx, thread_axis("blockIdx.x"))
    s[op].bind(tx, thread_axis(axis_name))

    return tx


@schedule_complex_reduce.register('gpu')
def schedule_complex_reduce_gpu(s, cfg, node):
    op = node.op
    output = node.compute_at_loc.op

    n_sp = len(s[op].op.axis)
    n_rd = len(s[op].op.reduce_axis)

    # ==================== Define Configuration Space ====================
    prefix = "#2-" + node.name
    tile_level = 4
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        sp_chains = []
        for i, axis in enumerate(s[op].op.axis):
            ch = cfg.define_split(prefix + "_sp_tile_" + str(i), _get_axis_length(axis),
                                  num_outputs=tile_level)
            sp_chains.append(ch)

        rd_chains = []
        for i, axis in enumerate(s[op].op.reduce_axis):
            ch = cfg.define_split(prefix + "_rd_tile_" + str(i), _get_axis_length(axis),
                                  num_outputs=tile_level-2)
            rd_chains.append(ch)



    # ========================== Heuristic Fill ==========================

    # =========================== Apply Config ===========================
    input_tensor = s[op].op.input_tensors
    tensor = s[op].output(0)
    shared_cache = []

    # add cache read and write stage
    for buf in input_tensor:
        shared_cache.append(s.cache_read(buf, "shared", [buf]))

    if op == output:
        local = s.cache_write(tensor, "local")
        output = tensor
    else:
        local = tensor
        output = output

@schedule_simple_reduce.register('gpu')
def schedule_simple_reduce_gpu(s, cfg, node):
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
        spatial_prod = np.prod([_get_axis_length(x) for x in s[op].op.axis])
        reduction_prod = np.prod([_get_axis_length(x) for x in s[op].op.reduce_axis])

        if spatial_prod >= _get_axis_length(s[op].op.reduce_axis[0]) \
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
        s[T].bind(tx, thread_axis('threadIdx.x'))
        s[TF].compute_at(s[T], tx)

        s[output].set_store_predicate(thread_x.equal(0))
        if len(s[T].op.axis) != 0:
            num_tx = _target.current_target().max_num_threads // cfg[prefix + '_parallel_num'].val
            return _parallel_spatial(s, output, s[output].op.axis, num_tx, "threadIdx.y")
        else:
            return s[output].op.axis[-1]

    tensor = op.output(0)

    if cfg[prefix + '_parallel_loc'].val == 'spatial' and False:
        last = _parallel_spatial(s, output, s[output].op.axis)
    else:
        last = _parallel_reduction(tensor, output, cfg[prefix + '_parallel_num'].val)

    if op != output:
        s[op].compute_at(s[output], last)


@schedule_other_root.register('gpu')
def schedule_other_root_gpu(s, cfg, node):
    op = node.op

    # todo(lmzheng): reordering?
    _parallel_spatial(s, op, s[op].op.axis)
