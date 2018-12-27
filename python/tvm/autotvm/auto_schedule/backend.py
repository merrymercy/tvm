
import numpy as np

from ... import target as _target, expr as _expr, tensor as _tensor, thread_axis, ir_pass, schedule
from ..task.space import get_factors
from .common import AutoScheduleOptions as opts, _get_axis_length

@_target.generic_func
def schedule_complex_reduce(s, node):
    raise NotImplementedError()

@_target.generic_func
def schedule_simple_reduce(s, node):
    raise NotImplementedError()

@_target.generic_func
def schedule_other_root(s, node):
    raise NotImplementedError()

@schedule_complex_reduce.register('gpu')
def schedule_complex_reduce_gpu(s, node):
    pass

@schedule_simple_reduce.register('gpu')
def schedule_simple_reduce_gpu(s, node):
    op = node.op
    output = node.compute_at_loc.op

    spatial_prod = np.prod([_get_axis_length(x) for x in s[op].op.axis])
    reduction_prod = np.prod([_get_axis_length(x) for x in s[op].op.reduce_axis])


@schedule_other_root.register('gpu')
def schedule_other_root_gpu(s, node):
    op = node.op
    axes = s[op].op.axis

    # todo(lmzheng): reordering?

    max_thread = _target.current_target().max_num_threads
    prod = 1

    tx = None
    i = 0
    for i, axis in enumerate(reversed(axes)):
        if prod * _get_axis_length(axis) > max_thread:
            break
        tx = axis if tx is None else s[op].fuse(axis, tx)
        prod = prod * _get_axis_length(axis)

    if tx is None:
        f = 1
        for f in get_factors(_get_axis_length(axes[-1])):
            if f <= max_thread:
                break
        f = max(max_thread // 2, f)
        outer, tx = s[op].split(axes[-1], f)
        bx = s[op].fuse(*(axes[:-1] + [outer]))
    else:
        bx = s[op].fuse(*axes[:-i])

    s[op].bind(bx, thread_axis("blockIdx.x"))
    s[op].bind(tx, thread_axis("threadIdx.x"))


def _parallel_cpu(s, op, axes):
    """parallel axes for cpu"""
    axis_parallel = axes[0]
    prod = _get_axis_length(axes[0])
    for axis in axes[1:]:
        if prod * _get_axis_length(axis) > opts.PARALLEL_THRESHOLD:
            break
        prod *= _get_axis_length(axis)
        axis_parallel = s[op].fuse(axis_parallel, axis)
    s[op].parallel(axis_parallel)


def _var_in_expr(var, expr):
    """check whether a variable is in an expr"""
    find = []

    def _check(stmt):
        if isinstance(stmt, _expr.Var) and stmt.same_as(var):
            find.append(True)

    ir_pass.PostOrderVisit(expr, _check)
    return bool(find)


def _is_strict_vectorizable(node, var):
    """check whether an expr is strictly vectorizable on one dimension"""
    if _get_axis_length(var) < opts.VEC_SIZE:
        return False

    for edge in node.read_edges.values():
        for pattern in edge.access:
            if len(pattern) == 0:
                continue
            if (any([_var_in_expr(var, expr) for expr in pattern[:-1]]) or
                    ((_var_in_expr(var, pattern[-1])) and not (pattern[-1].same_as(var)))):
                return False

    return True

@schedule_simple_reduce.register('cpu')
def schedule_simple_reduce_cpu(s, node):
    op = node.op
    output = node.compute_at_loc.op

    spatial_prod = np.prod([_get_axis_length(x) for x in s[op].op.axis])
    reduction_prod = np.prod([_get_axis_length(x) for x in s[op].op.reduce_axis])

    def _parallel_reduction(T):
        r = s[T].op.reduce_axis[0]
        ro, ri = s[T].split(r, nparts=opts.NUM_THREADS)
        TF = s.rfactor(T, ro, factor_axis=1)
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        s[TF].parallel(s[TF].op.axis[-1])
        other_reduce = list(s[TF].op.reduce_axis[:-1])
        inner = [s[TF].op.reduce_axis[-1]]
        s[TF].reorder(*(inner + other_reduce))

        s[TF].compute_at(s[T], s[T].op.axis[-1])

    def _vectorize_reduction(T):
        r = s[T].op.reduce_axis[-1]
        ro, ri = s[T].split(r, opts.VEC_SIZE)
        TF = s.rfactor(T, ri, factor_axis=len(s[T].op.axis))
        TF = TF[0] if not isinstance(TF, _tensor.Tensor) else TF

        s[TF].vectorize(s[TF].op.axis[-1])
        s[TF].reorder(*(list(s[TF].op.reduce_axis) + [s[TF].op.axis[-1]]))

        s[TF].compute_at(s[T], s[T].op.axis[-1])

    def _parallel_vectorize_reduction(T):
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

    tensor = s[op].op.output(0)

    spaital_parallel_flag = spatial_prod > opts.NUM_THREADS
    reduction_parallel_flag = _get_axis_length(s[op].op.axis[0]) >= opts.NUM_THREADS * 16
    reduction_vectorize_flag = _is_strict_vectorizable(node, s[op].op.axis[-1])

    if spaital_parallel_flag:
        if reduction_vectorize_flag:
            _vectorize_reduction(tensor)
        _parallel_cpu(s, output, s[output].op.axis)
    else:
        if len(s[op].op.reduce_axis) >= 2:
            if reduction_parallel_flag and reduction_vectorize_flag:     # tunable ?
                _parallel_vectorize_reduction(tensor)
            elif reduction_parallel_flag:
                _parallel_reduction(tensor)
            elif reduction_vectorize_flag:
                _vectorize_reduction(tensor)
            else:
                _parallel_cpu(s, output, s[op].op.axis)
        else:
            if reduction_parallel_flag:                        # tunable ?
                _parallel_reduction(tensor)
            elif reduction_vectorize_flag:
                _vectorize_reduction(tensor)
            else:
                _parallel_cpu(s, output, s[output].op.axis)

    if op != output:
        s[op].compute_at(s[output], s[output].op.axis[-1])


@schedule_other_root.register('cpu')
def schedule_other_root_cpu(s, node):
    op = node.op
    axes = s[op].op.axis

    # check the possibility of vectorization
    outer, vec = s[op].split(axes[-1], opts.VEC_SIZE)
    s[op].vectorize(vec)

    _parallel_cpu(s, op, list(axes)[:-1] + [outer])
