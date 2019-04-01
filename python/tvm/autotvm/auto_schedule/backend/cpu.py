"""Schedule template for CPU"""

import numpy as np

from .... import expr as _expr, tensor as _tensor, ir_pass as _ir_pass
from ...task.space import get_factors, ConfigSpace, FallbackConfigEntity
from ..stage_analysis import StageEdgeType
from ..common import AutoScheduleOptions as opts, get_axis_length, tuning_level
from .generic import schedule_other_root, schedule_simple_reduce, \
    schedule_complex_reduce, schedule_tune_direct_compute


def _parallel_spatial(s, op, axes):
    """parallelize spatial axes for cpu"""
    if len(axes) == 0:
        return None

    axis_parallel = axes[0]
    prod = get_axis_length(axes[0])

    last = None
    for axis in axes[1:]:
        if prod % opts.NUM_THREADS == 0:
            last = axis
            break
        prod *= get_axis_length(axis)
        axis_parallel = s[op].fuse(axis_parallel, axis)

    s[op].parallel(axis_parallel)
    return axis_parallel if last is None else last


def _var_in_expr(var, expr):
    """check whether a variable is in an expr"""
    find = []

    def _check(stmt):
        if isinstance(stmt, _expr.Var) and stmt.same_as(var):
            find.append(True)

    _ir_pass.PostOrderVisit(expr, _check)
    return bool(find)

def _divisible_split(s, T, axis, factor=None, nparts=None):
    """Split an axis only by divisible factors. This is a walk around for partial tiles."""
    # todo(lmzheng): remove this after fix loop partition
    assert nparts is None or factor is None
    if nparts is not None:
        nparts = [x for x in get_factors(get_axis_length(axis)) if x >= nparts][0]
        return s[T].split(axis, nparts=nparts)
    if factor is not None:
        factor = [x for x in get_factors(get_axis_length(axis)) if x <= factor][-1]
        return s[T].split(axis, factor=factor)


def _is_vectorizable(node, axis, strict=True):
    """check whether an expr is vectorizable on one dimension.
    strict condition: For all read accesses, this axis can only appear directly as the last dimension.
    relaxed condition: Allow some invalid access patterns, but the number of valid access should be
                       greater than that of invalid ones.
    """
    if get_axis_length(axis) < opts.VEC_SIZE:
        return False

    success = 0
    fail = 0

    var = axis.var
    for edge in node.read_edges.values():
        for pattern in edge.access:
            if len(pattern) == 0:
                continue
            if (all([not _var_in_expr(var, expr) for expr in pattern[:-1]]) and
                    (not(_var_in_expr(var, pattern[-1])) or (pattern[-1].same_as(var)))):
                success += 1
            else:
                fail += 1

    if strict:
        return fail == 0
    else:
        return success >= fail

def _schedule_root(s, op, axes, do_vec):
    """Schedule root nodes for cpu.
    Vectorize the innermost axis and parallelize the outermost axis"""
    if do_vec and get_axis_length(axes[-1]) >= opts.VEC_SIZE:
        last = _parallel_spatial(s, op, axes)
        par, vec = _divisible_split(s, op, last, opts.VEC_SIZE)
        s[op].vectorize(vec)
        if last not in s[op].op.axis:
            s[op].parallel(par)
    else:
        _parallel_spatial(s, op, axes)


@schedule_tune_direct_compute.register('cpu')
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
    dst = node.compute_at_loc
    axes = s[op].op.axis

    target_axes = s[dst.op].leaf_iter_vars
    target_axes = target_axes[::2]  # prune some locations

    # ==================== Define Configuration Space ====================
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        cfg.define_knob("#2-" + node.name + '_compute_axis', list(range(-2, len(target_axes))))
        cfg.define_knob("#3-" + node.name + '_vec', [False, True])

    # ========================== Heuristic Fill ==========================
    if tuning_level(cfg) < 2:
        if all([x.type in [StageEdgeType.ELEMWISE, StageEdgeType.BROADCAST]
                for x in node.read_edges.values()]):
            cfg["#2-" + node.name + '_compute_axis'].val = -2
        else:
            cfg["#2-" + node.name + '_compute_axis'].val = -1
    if tuning_level(cfg) < 3:
        if len(axes) > 0:
            cfg["#3-" + node.name + '_vec'].val = _is_vectorizable(node, axes[-1])

    # =========================== Apply Config ===========================
    val = cfg["#2-" + node.name + '_compute_axis'].val
    if val == -1:
        _schedule_root(s, op, axes, cfg["#3-" + node.name + '_vec'].val)
    elif val == -2:
        s[op].compute_inline()
    else:
        s[op].compute_at(s[dst.op], target_axes[val])
        if cfg["#3-" + node.name + '_vec'].val and \
                        get_axis_length(axes[-1]) >= opts.VEC_SIZE:
            # todo(lmzheg): remove this condition after fix loop partition
            _, vec = _divisible_split(s, op, axes[-1], opts.VEC_SIZE)
            s[op].vectorize(vec)


@schedule_other_root.register('cpu')
def schedule_other_root_cpu(s, cfg, node):
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
    axes = s[op].op.axis

    # ==================== Define Configuration Space ====================
    prefix = "#3-" + node.name
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        cfg.define_knob(prefix + '_vec', [True, False])

    # ========================== Heuristic Fill ==========================
    if tuning_level(cfg) < 3:
        cfg[prefix + '_vec'].val = _is_vectorizable(node, axes[-1])

    # =========================== Apply Config ===========================
    _schedule_root(s, op, axes, cfg[prefix + '_vec'].val)


@schedule_simple_reduce.register('cpu')
def schedule_simple_reduce_cpu(s, cfg, node):
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
        # whether vectorize reduction axis
        cfg.define_knob(prefix + '_vec_reduction', [True, False])

    # ========================== Heuristic Fill ==========================
    if tuning_level(cfg) < 2:
        spatial_prod = np.prod([get_axis_length(x) for x in s[op].op.axis])
        reduction_prod = np.prod([get_axis_length(x) for x in s[op].op.reduce_axis])
        # todo(lmzheng) : fuse reduction axis

        if spatial_prod > opts.NUM_THREADS \
                or get_axis_length(s[op].op.reduce_axis[0]) < opts.NUM_THREADS * 4:
            cfg[prefix + '_parallel_loc'].val = 'spatial'
            cfg[prefix + '_vec_reduction'].val = _is_vectorizable(node,
                                                                  s[op].op.reduce_axis[-1])
        else:
            cfg[prefix + '_parallel_loc'].val = 'reduction'
            cfg[prefix + '_vec_reduction'].val = _is_vectorizable(node, s[op].op.reduce_axis[-1]) \
                                                 and reduction_prod > opts.NUM_THREADS * opts.VEC_SIZE

    # =========================== Apply Config ===========================
    # Here we have three kinds of strategies:
    # 1. parallelize reduction axis
    # 2. vectorize reduction axis
    # 3. do both, parallel and vectorize reduction axis
    def _parallel_reduction(T):
        r = s[T].op.reduce_axis[0]
        #ro, ri = s[T].split(r, nparts=opts.NUM_THREADS)
        ro, ri = _divisible_split(s, T, r, nparts=opts.NUM_THREADS)
        TF = s.rfactor(T, ro, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        s[TF].parallel(s[TF].op.axis[-1])
        other_reduce = list(s[TF].op.reduce_axis[:-1])
        inner = [s[TF].op.reduce_axis[-1]]
        s[TF].reorder(*(inner + other_reduce))

        s[TF].compute_at(s[T], s[T].op.axis[-1])
        return TF

    def _vectorize_reduction(T):
        r = s[T].op.reduce_axis[-1]
        #ro, ri = s[T].split(r, opts.VEC_SIZE)
        ro, ri = _divisible_split(s, T, r, opts.VEC_SIZE)
        TF = s.rfactor(T, ri, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        s[TF].vectorize(s[TF].op.axis[-1])
        s[TF].reorder(*(list(s[TF].op.reduce_axis) + [s[TF].op.axis[-1]]))

        if len(s[T].op.axis) >= 1:
            s[TF].compute_at(s[T], s[T].op.axis[-1])
        return TF

    def _parallel_vectorize_reduction_single(T):
        # parallel
        r = s[T].op.reduce_axis[0]
        #ro, ri = s[T].split(r, nparts=opts.NUM_THREADS)
        ro, ri = _divisible_split(s, T, r, nparts=opts.NUM_THREADS)
        TF = s.rfactor(T, ro, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        # vectorize
        r = s[TF].op.reduce_axis[0]
        #ro, ri = s[TF].split(r, opts.VEC_SIZE)
        ro, ri = _divisible_split(s, TF, r, opts.VEC_SIZE)
        TFF = s.rfactor(TF, ri, factor_axis=len(s[TF].op.axis))
        TFF = TFF[0] if not isinstance(TFF, _tensor.Tensor) else TFF

        s[TF].parallel(s[TF].op.axis[-1])
        s[TFF].vectorize(s[TFF].op.axis[-1])

        # fuse
        s[TF].compute_at(s[T], s[T].op.axis[-1])
        s[TFF].compute_at(s[TF], s[TF].op.axis[-1])

    def _parallel_vectorize_reduction_multi(T):
        # parallel
        r = s[T].op.reduce_axis[0]
        #ro, ri = s[T].split(r, nparts=opts.NUM_THREADS)
        ro, ri = _divisible_split(s, T, r, nparts=opts.NUM_THREADS)
        TF = s.rfactor(T, ro, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        # vectorize
        r = s[TF].op.reduce_axis[-2]
        #ro, ri = s[TF].split(r, opts.VEC_SIZE)
        ro, ri = _divisible_split(s, TF, r, opts.VEC_SIZE)
        TFF = s.rfactor(TF, ri, factor_axis=len(s[TF].op.axis))
        TFF = TFF[0] if not isinstance(TFF, _tensor.Tensor) else TFF

        s[TF].parallel(s[TF].op.axis[-1])
        s[TFF].vectorize(s[TFF].op.axis[-1])

        # reorder
        innermost = s[TFF].op.axis[-1]
        outermost = s[TFF].op.reduce_axis[-2]
        other_reduce = list(s[TFF].op.reduce_axis)
        other_reduce.remove(outermost)
        s[TFF].reorder(*([outermost] + other_reduce + [innermost]))

        # fuse
        if len(s[T].op.axis) >= 1:
            s[TF].compute_at(s[T], s[T].op.axis[-1])
        if len(s[TF].op.axis) >= 1:
            s[TFF].compute_at(s[TF], s[TF].op.axis[-1])

    tensor = op.output(0)

    last = None
    if cfg[prefix + '_parallel_loc'].val == 'reduction':
        if cfg[prefix + '_vec_reduction'].val:
            if len(s[op].op.reduce_axis) == 1:
                _parallel_vectorize_reduction_single(tensor)
            else:
                _parallel_vectorize_reduction_multi(tensor)
        else:
            _parallel_reduction(tensor)

        if len(s[output].op.axis) >= 1:
            last = s[output].op.axis[-1]
    else:
        if cfg[prefix + '_vec_reduction'].val:
            TF = _vectorize_reduction(tensor)
            last = _parallel_spatial(s, output, s[output].op.axis)
            if last is not None:
                s[TF].compute_at(s[output], last)
        else:
            last = _parallel_spatial(s, output, s[output].op.axis)

    if op != output and last is not None:
        s[op].compute_at(s[output], last)


@schedule_complex_reduce.register('cpu')
def schedule_complex_reduce_cpu(s, cfg, node):
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

    n_sp = len(s[op].op.axis)
    n_rd = len(s[op].op.reduce_axis)
    n_sp_ann = min(3, n_sp)
    n_rd_ann = min(2, n_rd)

    # ==================== Define Configuration Space ====================
    prefix = "#1-" + node.name
    tile_level = 3
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        sp_chains = []
        for i, axis in enumerate(s[op].op.axis):
            ch = cfg.define_split(prefix + "_sp_tile_%d" % i, get_axis_length(axis),
                                  num_outputs=tile_level)
            sp_chains.append(ch)

        rd_chains = []
        for i, axis in enumerate(s[op].op.reduce_axis):
            ch = cfg.define_split(prefix + "_rd_tile_%d" % i, get_axis_length(axis),
                                  num_outputs=tile_level-1)
            rd_chains.append(ch)

        cfg.define_annotate(prefix + "_ann_spatial",
                            [ch[-1] for ch in sp_chains[-n_sp_ann:]], policy='try_unroll_vec')
        cfg.define_annotate(prefix + "_ann_reduce",
                            [ch[-1] for ch in rd_chains[-n_rd_ann:]], policy='try_unroll')

    # ========================== Heuristic Fill ==========================
    if tuning_level(cfg) < 1:
        for i in range(n_sp_ann):
            cfg[prefix + "_ann_spatial"].anns[i] = "none"
        for i in range(n_rd_ann):
            cfg[prefix + "_ann_reduce"].anns[i] = "none"

        vec_able = _is_vectorizable(node, s[op].op.axis[-1], strict=False)
        if vec_able:
            cfg[prefix + "_ann_spatial"].anns[-1] = "vec"
        else:
            cfg[prefix + "_ann_spatial"].anns[-1] = "unroll"

        for i, axis in enumerate(s[op].op.reduce_axis):
            cfg[prefix + "_rd_tile_%d" % i].size = [get_axis_length(axis), 1]

        # tile only on last two dimensions
        ndim = min(2, n_sp)
        constraints = [1] * (n_sp - ndim) + [opts.TILE_SIZE] * ndim
        if vec_able:
            constraints[-1] = opts.VEC_SIZE

        for i, axis in enumerate(s[op].op.axis):
            length = get_axis_length(axis)
            inner = [x for x in get_factors(length) if x <= constraints[i]][-1]
            outer = length // inner
            cfg[prefix + "_sp_tile_%d" % i].size = [outer, 1, inner]

    # =========================== Apply Config ===========================
    sp_chains = []
    for i, axis in enumerate(s[op].op.axis):
        ch = cfg[prefix + "_sp_tile_%d" % i].apply(s, op, axis)
        sp_chains.append(ch)

    rd_chains = []
    for i, axis in enumerate(s[op].op.reduce_axis):
        ch = cfg[prefix + "_rd_tile_%d" % i].apply(s, op, axis)
        rd_chains.append(ch)

    cfg[prefix + "_ann_spatial"].apply(s, op, [ch[-1] for ch in sp_chains[-n_sp_ann:]],
                                       max_unroll=opts.MAX_UNROLL,
                                       axis_lens=[cfg[prefix + "_sp_tile_%d" % i].size[-1]
                                                  for i in range(n_sp)])
    cfg[prefix + "_ann_reduce"].apply(s, op, [ch[-1] for ch in rd_chains[-n_rd_ann:]],
                                      max_unroll=opts.MAX_UNROLL,
                                      axis_lens=[cfg[prefix + "_rd_tile_%d" % i].size[-1]
                                                 for i in range(n_rd)])

    # reorder
    all_chains = sp_chains + rd_chains
    all_axes = []
    for i in range(tile_level):
        for ch in all_chains:
            if i < len(ch):
                all_axes.append(ch[i])

    s[op].reorder(*all_axes)

    if op != output:
        # apply operation of spatial axis to outer stage
        sp_chains = []
        for i, axis in enumerate(s[output].op.axis):
            ch = cfg[prefix + "_sp_tile_%d" % i].apply(s, output, axis)
            sp_chains.append(ch)
        # reorder
        all_axes = []
        for i in range(tile_level):
            for ch in sp_chains:
                all_axes.append(ch[i])
        s[output].reorder(*all_axes)
        # apply the same annotation to inner axes
        cfg[prefix + "_ann_spatial"].apply(s, output, all_axes[-n_sp_ann:],
                                           max_unroll=opts.MAX_UNROLL,
                                           axis_lens=[cfg[prefix + "_sp_tile_%d" % i].size[-1]
                                                      for i in range(n_sp - n_sp_ann, n_sp)])

        s[op].compute_at(s[output], all_axes[n_sp])
        final = output
    else:
        final = op

    parallel_axes = all_axes[:n_sp]
    # attach length information for newly created axes
    for i, axis in enumerate(parallel_axes):
        axis.attached_length = cfg[prefix + "_sp_tile_%d" % i].size[0]

    _parallel_spatial(s, final, parallel_axes)
