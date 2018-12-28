import numpy as np

from .... import target as _target, expr as _expr, tensor as _tensor, ir_pass
from ...task.space import get_factors, ConfigEntity, ConfigSpace, FallbackConfigEntity
from ..common import AutoScheduleOptions as opts, _get_axis_length
from .generic import schedule_other_root, schedule_simple_reduce, schedule_complex_reduce


def _parallel_spatial(s, op, axes):
    """parallel axes for cpu"""
    axis_parallel = axes[0]
    prod = _get_axis_length(axes[0])

    last = None
    for axis in axes[1:]:
        if prod % opts.NUM_THREADS == 0:
            last = axis
            break
        prod *= _get_axis_length(axis)
        axis_parallel = s[op].fuse(axis_parallel, axis)

    s[op].parallel(axis_parallel)
    return axis_parallel if last is None else last


def _var_in_expr(var, expr):
    """check whether a variable is in an expr"""
    find = []

    def _check(stmt):
        if isinstance(stmt, _expr.Var) and stmt.same_as(var):
            find.append(True)

    ir_pass.PostOrderVisit(expr, _check)
    return bool(find)


def _is_strict_vectorizable(node, axis):
    """check whether an expr is strictly vectorizable on one dimension"""
    if _get_axis_length(axis) < opts.VEC_SIZE:
        return False

    var = axis.var
    for edge in node.read_edges.values():
        for pattern in edge.access:
            if len(pattern) == 0:
                continue
            if (any([_var_in_expr(var, expr) for expr in pattern[:-1]]) or
                    ((_var_in_expr(var, pattern[-1])) and not (pattern[-1].same_as(var)))):
                return False

    return True


@schedule_simple_reduce.register('cpu')
def schedule_simple_reduce_cpu(s, cfg, node):
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
    if isinstance(cfg, FallbackConfigEntity) or opts.TUNING_LEVEL < 2:
        spatial_prod = np.prod([_get_axis_length(x) for x in s[op].op.axis])
        reduction_prod = np.prod([_get_axis_length(x) for x in s[op].op.reduce_axis])
        # todo(lmzheng) : fuse reduction axis

        if spatial_prod > opts.NUM_THREADS\
                or _get_axis_length(s[op].op.reduce_axis[0]) < opts.NUM_THREADS * 4:
            cfg[prefix + '_parallel_loc'].val = 'spatial'
            cfg[prefix + '_vec_reduction'].val = _is_strict_vectorizable(node,
                                                                         s[op].op.reduce_axis[-1])
        else:
            cfg[prefix + '_parallel_loc'].val = 'reduction'
            cfg[prefix + '_vec_reduction'].val = _is_strict_vectorizable(node, s[op].op.reduce_axis[-1])\
                and reduction_prod > opts.NUM_THREADS * opts.VEC_SIZE

    # =========================== Apply Config ===========================
    def _parallel_reduction(T):
        r = s[T].op.reduce_axis[0]
        ro, ri = s[T].split(r, nparts=opts.NUM_THREADS)
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
        ro, ri = s[T].split(r, opts.VEC_SIZE)
        TF = s.rfactor(T, ri, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        s[TF].vectorize(s[TF].op.axis[-1])
        s[TF].reorder(*(list(s[TF].op.reduce_axis) + [s[TF].op.axis[-1]]))

        s[TF].compute_at(s[T], s[T].op.axis[-1])
        return TF

    def _parallel_vectorize_reduction_multi(T):
        # parallel
        r = s[T].op.reduce_axis[0]
        ro, ri = s[T].split(r, nparts=opts.NUM_THREADS)
        TF = s.rfactor(T, ro, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        # vectorize
        r = s[TF].op.reduce_axis[-2]
        ro, ri = s[TF].split(r, opts.VEC_SIZE)
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
        s[TF].compute_at(s[T], s[T].op.axis[-1])
        s[TFF].compute_at(s[TF], s[TF].op.axis[-1])

    def _parallel_vectorize_reduction_single(T):
        # parallel
        r = s[T].op.reduce_axis[0]
        ro, ri = s[T].split(r, nparts=opts.NUM_THREADS)
        TF = s.rfactor(T, ro, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        # vectorize
        r = s[TF].op.reduce_axis[0]
        ro, ri = s[TF].split(r, opts.VEC_SIZE)
        TFF = s.rfactor(TF, ri, factor_axis=len(s[TF].op.axis))
        TFF = TFF[0] if not isinstance(TFF, _tensor.Tensor) else TFF

        s[TF].parallel(s[TF].op.axis[-1])
        s[TFF].vectorize(s[TFF].op.axis[-1])

        # fuse
        s[TF].compute_at(s[T], s[T].op.axis[-1])
        s[TFF].compute_at(s[TF], s[TF].op.axis[-1])

    tensor = op.output(0)

    if cfg[prefix + '_parallel_loc'].val == 'reduction':
        if cfg[prefix + '_vec_reduction'].val:
            if len(s[op].op.reduce_axis) == 1:
                _parallel_vectorize_reduction_single(tensor)
            else:
                _parallel_vectorize_reduction_multi(tensor)
        else:
            _parallel_reduction(tensor)
        last = s[output].op.axis[-1]
    else:
        if cfg[prefix + '_vec_reduction'].val:
            TF = _vectorize_reduction(tensor)
            last = _parallel_spatial(s, output, s[output].op.axis)
            s[TF].compute_at(s[output], last)
        else:
            last = _parallel_spatial(s, output, s[output].op.axis)

    if op != output:
        s[op].compute_at(s[output], last)


@schedule_complex_reduce.register('cpu')
def schedule_complex_reduce_cpu(s, cfg, node):
    op = node.op
    output = node.compute_at_loc.op

    # ==================== Define Configuration Space ====================
    prefix = "#1-" + node.name
    tile_level = 3
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        sp_chains = []
        for i, axis in enumerate(s[op].op.axis):
            ch = cfg.define_split(prefix + "_sp_tile_" + str(i), _get_axis_length(axis),
                                  num_outputs=tile_level)
            sp_chains.append(ch)

        rd_chains = []
        for i, axis in enumerate(s[op].op.reduce_axis):
            ch = cfg.define_split(prefix + "_rd_tile_" + str(i), _get_axis_length(axis),
                                  num_outputs=tile_level)
            rd_chains.append(ch)

        cfg.define_annotate("ann_spatial", [ch[-1] for ch in sp_chains], policy='try_unroll_vec')
        cfg.define_annotate("ann_reduce", [ch[-1] for ch in rd_chains], policy='try_unroll')

    # ========================== Heuristic Fill ==========================
    # if isinstance(cfg, FallbackConfigEntity) or opts.TUNING_LEVEL < 1:
    #     cfg[prefix + '_vec'].val = _is_strict_vectorizable(node, axes[-1])

    # =========================== Apply Config ===========================
    sp_chains = []
    for i, axis in enumerate(s[op].op.axis):
        ch = cfg[prefix + "_sp_tile_" + str(i)].apply(s, op, axis)
        sp_chains.append(ch)

    rd_chains = []
    for i, axis in enumerate(s[op].op.reduce_axis):
        ch = cfg[prefix + "_rd_tile_" + str(i)].apply(s, op, axis)
        rd_chains.append(ch)

    cfg["ann_spatial"].apply(s, op, [ch[-1] for ch in sp_chains],
                             max_unroll=opts.MAX_UNROLL,
                             axis_lens=[cfg[prefix + "_sp_tile_" + str(i)].size[-1]
                                        for i in range(len(s[op].op.axis))])
    cfg["ann_reduce"].apply(s, op, [ch[-1] for ch in rd_chains],
                            max_unroll=opts.MAX_UNROLL,
                            axis_lens=[cfg[prefix + "_rd_tile_" + str(i)].size[-1]
                                       for i in range(len(s[op].op.reduce_axis))])

    # reorder
    all_chains = sp_chains + rd_chains
    all_axes = []
    for i in range(tile_level):
        for ch in all_chains:
            all_axes.append(ch[i])

    s[op].reorder(*all_axes)

    if op != output:
        # apply operation of spatial axis to outer stage
        sp_chains = []
        for i, axis in enumerate(s[output].op.axis):
            ch = cfg[prefix + "_sp_tile_" + str(i)].apply(s, output, axis)
            sp_chains.append(ch)
        all_axes = []
        for i in range(tile_level):
            for ch in sp_chains:
                all_axes.append(ch[i])
        s[output].reorder(*all_axes)

        s[op].compute_at(s[output], all_axes[len(s[output].op.axis)])


@schedule_other_root.register('cpu')
def schedule_other_root_cpu(s, cfg, node):
    op = node.op
    axes = s[op].op.axis

    # ==================== Define Configuration Space ====================
    prefix = "#3-" + node.name
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        cfg.define_knob(prefix + '_vec', [True, False])

    # ========================== Heuristic Fill ==========================
    if isinstance(cfg, FallbackConfigEntity) or opts.TUNING_LEVEL < 3:
        cfg[prefix + '_vec'].val = _is_strict_vectorizable(node, axes[-1])

    # =========================== Apply Config ===========================
    if cfg[prefix + '_vec'].val:
        last = _parallel_spatial(s, op, axes)
        par, vec = s[op].split(last, opts.VEC_SIZE)
        s[op].vectorize(vec)
        if last not in s[op].op.axis:
            s[op].parallel(par)
    else:
        _parallel_spatial(s, op, axes)
