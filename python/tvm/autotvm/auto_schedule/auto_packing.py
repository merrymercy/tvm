"""
Pack tensors (transform layout) to better utilize vectorization and increase memory locality.
This is for cpu only.

Example:
C[i, j] = sum(A[i, k] * B[j, k], k)

will be transformed to

A_pack[i, j, ii] = A[i * 16 + ii, j]
C_pack[i, j, ii] = sum(A[i, k, ii] * B_pack[j, k], k)
C[i, j] = C_pack[i/16, j, i%16]
"""


import math
from functools import reduce
from collections import namedtuple

import numpy as np

from ... import tensor as _tensor, expr as _expr, target as _target, \
    make as _make, schedule as _schedule, arith as _arith, _api_internal,\
    bijective_layout, ir_pass, compute, Range
from ..util import get_const_tuple
from ..task.space import get_factors, ConfigSpace, FallbackConfigEntity
from .stage_analysis import topo_order, _gather_vars, StageNodeType
from .backend.cpu import _is_vectorizable
from .common import AutoScheduleOptions as opts, get_axis_length, tuning_level

# use op + index to represent the axis `op.axis[index]`
IndexAxis = namedtuple("IndexAxis", ['op', 'index'])

def _node_index(array, x):
    """Get the index of x in array. But use NodeBase::same_as as equality test"""
    for i, y in enumerate(array):
        if x.same_as(y):
            return i
    return -1


def _find_vectorizable_axes(n):
    """Condition: For a spatial axis x, all index expressions contain x must only contain the single
    variable x (not 2 * x or x + y). Though this is still too strict.

    e.g. A[i, j, k] = C[i * 2, j, k/2]
    Then j is considered vectorizable while i and k are not.
    """
    bad_axes = set()
    good_axes = set()

    for edge in n.read_edges.values():
        for pattern in edge.access:
            for expr in pattern:
                if isinstance(expr, _expr.Var):
                    good_axes.add(expr)
                else:
                    bad_vars = _gather_vars(expr)
                    bad_axes.update(bad_vars)

    ret = []
    for i, axis in enumerate(n.op.axis):
        v = axis.var
        if v in bad_axes or v not in good_axes:
            continue
        ret.append((n, i))

    return ret


def detect_linked_axes(node_dict):
    """Detect linked axes in Stage Graph.

    Parameters
    ----------
    node_dict: dict[op -> StageNode]
        The dict represents stage graph

    Returns
    -------
    linked_axes: dict[IndexAxis -> set of IndexAxis]
        All linked axes will be gathered into a set.
        This dict maps a axis to the set that contains it.
    """

    # Find linked axes, their layouts can be tied
    # For tensor S and T, assume S reads T. Let x be a spatial axis of S, y be a spatial axis of T.
    # If for all read accesses from S to T, x exactly matches dimension y, then we can link x and y.
    #
    # e.g. A[i, j] = C[i, j+1] + 1, then A.op.axis[0] and C.op.axis[0] are considered linked because all
    # read/write access between these two axes are elemwise.

    link_edges = []  # list of linked edges
    for n in topo_order(node_dict):
        if isinstance(n.op, _tensor.ComputeOp):
            for dst, edge in n.read_edges.items():
                sp_axis_vars = [x.var for x in n.op.axis]

                tmp_link_table = [[] for _ in range(len(sp_axis_vars))]
                fail_flag = [False] * len(sp_axis_vars)

                # for all access from node `n` to node `dst`
                for pattern in edge.access:
                    for i, expr in enumerate(pattern):
                        if isinstance(expr, _expr.Var):  # if is a direct access (linkable)
                            idx = _node_index(sp_axis_vars, expr)
                            if idx != -1:
                                tmp_link_table[idx].append(i)
                        else:                            # all axes in this index are not linkable
                            for x in _gather_vars(expr):
                                idx = _node_index(sp_axis_vars, x)
                                if idx != -1:
                                    fail_flag[idx] = True

                for i, linked in enumerate(tmp_link_table):
                    if fail_flag[i]:
                        continue
                    if len(linked) != 1:
                        continue
                    link_edges.append((IndexAxis(n, i), IndexAxis(dst, linked[0])))

    # use union set to union axes
    parent = {}
    def _find_parent(x):
        if parent[x] == x:
            return x
        else:
            y = _find_parent(parent[x])
            parent[x] = y
            return y

    def _union(x, y):
        parent[_find_parent(x)] = _find_parent(y)

    for u, v in link_edges:
        parent[u] = u
        parent[v] = v
    for u, v in link_edges:
        _union(u, v)
    for u, v in link_edges:
        a = _find_parent(u)
        b = _find_parent(v)
        assert a == b, "%s %s" % (a, b)

    # build the map axes -> linked_set
    linked_axes = {}   # IndexAxis -> The set contains all linked edges
    for u, v in link_edges:
        z = _find_parent(u)
        if z not in linked_axes:
            linked_axes[z] = set()
        add_to = linked_axes[z]
        linked_axes[u] = add_to
        linked_axes[v] = add_to
        add_to.add(u)
        add_to.add(v)

    # CHECK: for all axes in a linked set, they should belong to different buffer
    for link_set in linked_axes.values():
        for x in link_set:
            for y in link_set:
                if x != y:
                    assert x.op != y.op, "%s %s" % (x, y)

    return linked_axes

LAYOUT_STRING = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def plan_layout(node_dict, io_bufs, linked_axes, cfg):
    """Plan layout for every nodes.
    Current implementation considers the opportunity to vectorize.

    Parameters
    ----------
    node_dict: dict[op -> StageNode]
        The dict represents stage graph
    io_bufs: list of Tensors
        The input/output buffers, their layout cannot be changed
    linked_axes: dict[IndexAxis -> set of IndexAxis]
        All linked axes will be gathered into a set.
        This dict maps a axis to the set that contains it.
    cfg: ConfigEntity
        The current configuration for the template
    """
    # Pick what to split
    vectorizable_axes = []
    scores = []

    # Get all vectorizable axes
    for n in node_dict.values():
        if n.type == StageNodeType.COMPLEX_REDUCTION:
            if _is_vectorizable(n, n.op.axis[-1]):
                continue
            vectorizable_axes.extend([x for x in _find_vectorizable_axes(n)
                                      if get_axis_length(n.op.axis[x[1]]) >= opts.VEC_SIZE])

    # Use a cost function to choose which axis to split
    # cost function: 5 * enabled vectorized FLOPs - data transform
    for node, idx in vectorizable_axes:
        link_set = linked_axes[(node, idx)]

        s = 0
        for node, _ in link_set:
            tensor = n.op.output(0)
            if n.type == StageNodeType.COMPLEX_REDUCTION:
                s += 5 * np.prod(get_const_tuple(tensor.shape))

            if n in io_bufs:
                s -= np.prod(get_const_tuple(tensor.shape))
        scores.append(s)

    rank = np.argsort(scores)[::-1]

    split_set = []
    already_vec = set()
    for x in rank:
        node, idx = vectorizable_axes[x]
        if node in already_vec:
            continue

        link_set = linked_axes[(node, idx)]
        lens = []
        for node, idx in link_set:
            already_vec.add(node)
            lens.append(node.op.output(0).shape[idx].value)

        gcd = reduce(math.gcd, lens)
        factors = [x for x in get_factors(gcd) if opts.VEC_SIZE //2 <= x <= 2 * opts.VEC_SIZE]
        if factors:
            split_set.append((link_set, factors))

    # ==================== Define Configuration Space ====================
    if isinstance(cfg, (ConfigSpace, FallbackConfigEntity)):
        for i, (link_set, factors) in enumerate(split_set):
            cfg.define_knob('#1-auto_pack_vec_%d' % i, factors + [opts.VEC_SIZE])

    # ========================== Heuristic Fill ==========================
    if tuning_level(cfg) < 1:
        for i, (link_set, factors) in enumerate(split_set):
            for j in reversed(range(len(factors))):
                if factors[j] < opts.VEC_SIZE:
                    break
            cfg['#1-auto_pack_vec_%d' % i].val = factors[min(j+1, len(factors)-1)]

    # =========================== Apply Config ===========================
    # assign new layout
    layout_dict = {}        # Tensor -> BijectiveLayout
    for i, (link_set, _) in enumerate(split_set):
        for node, idx in link_set:
            tensor = node.op.output(0)
            src_layout = LAYOUT_STRING[:len(tensor.shape)]
            dst_layout = src_layout + "%d" % cfg["#1-auto_pack_vec_%d" % i].val + src_layout[idx].lower()
            layout_dict[tensor] = bijective_layout(src_layout, dst_layout)

    for op in node_dict:
        tensor = op.output(0)
        if tensor in layout_dict:
            continue
        src_layout = LAYOUT_STRING[:len(tensor.shape)]
        layout_dict[tensor] = bijective_layout(src_layout, src_layout)

    return layout_dict, bool(split_set)

def _replace_layout(expr, var_dict, layout_dict, replace_dict, dom_dict):
    """Replace layout for an expr"""
    analyzer = _arith.Analyzer()
    for v, (low, high) in dom_dict:
        low, high = get_const_tuple((low, high))
        analyzer.update(v, _arith.ConstIntBound(low, high - 1))

    def replace_call(op):
        if isinstance(op, _expr.Call):
            if op.call_type == _expr.Call.Halide:
                tensor = op.func.output(0)
                trans = layout_dict.get(tensor, None)
                tensor = replace_dict.get(tensor, tensor)
                if trans:
                    args = trans.forward_index(op.args)
                else:
                    args = op.args

                return _make.Call(op.dtype, tensor.op.name, [analyzer.rewrite_simplify(x) for x in args],
                                  _expr.Call.Halide, tensor.op, 0)
        if isinstance(op, _expr.Var):
            return var_dict.get(op, None)

        return None

    stmt = _make.Evaluate(expr)
    stmt = ir_pass.IRTransform(stmt, None, replace_call, ['Call', 'Variable'])
    return stmt.value

def rewrite_compute(node_dict, io_bufs, layout_dict):
    """Rewrite compute according to the layout assignment

    Parameters
    ----------
    node_dict: dict[op -> StageNode]
        The parsed stage graph
    io_bufs: list of Tensor
        The input/output buffers
    layout_dict: dict[Tensor -> BijectiveLayout]
        The layout assignment for tensors

    Returns
    -------
    io_replace_dict: dict[Tensor -> Tensor]
        Dict for replacing input/output buffers
    """
    replace_dict = {}
    io_replace_dict = {}

    for node in topo_order(node_dict):
        tensor = node.op.output(0)
        layout_trans = layout_dict[tensor]

        if layout_trans.src_layout.name == layout_trans.dst_layout.name:
            pass

        if isinstance(node.op, _tensor.PlaceholderOp):
            new = compute(layout_trans.forward_shape(tensor.shape),
                          lambda *index: tensor(*layout_trans.backward_index(index)),
                          name=tensor.name + "_pack")
            replace_dict[tensor] = new
        elif isinstance(node.op, _tensor.ComputeOp):
            new_shape = layout_trans.forward_shape(tensor.shape)

            name_dict = {x: v.var.name for x, v in zip(layout_trans.src_layout.name, node.op.axis)}

            dim_vars = []
            dom_list = []
            for i, extent in enumerate(new_shape):
                layout_name = layout_trans.dst_layout[i]
                if layout_name.isupper():
                    name = name_dict[layout_name] + ".outer"
                else:
                    name = name_dict[layout_name.upper()] + '.inner'
                v = _api_internal._Var(name, 'int32')
                itervar = _api_internal._IterVar(Range(0, extent), v, _schedule.IterVar.DataPar, "")
                dim_vars.append(itervar)
                dom_list.append((v, (0, extent)))

            raw = [x.var for x in node.op.axis]
            transformed = layout_trans.backward_index([x.var for x in dim_vars])
            var_dict = {x: y for x, y in zip(raw, transformed)}

            body = _replace_layout(node.op.body[0], var_dict, layout_dict, replace_dict, dom_list)
            op = _api_internal._ComputeOp(tensor.name + "_pack", node.op.tag,
                                          node.op.attrs, dim_vars, [body])
            replace_dict[tensor] = op.output(0)

            if tensor in io_bufs:  # is an output buffer, unpack
                unpack_output = compute(tensor.shape,
                                        lambda *index: op.output(0)(*layout_trans.forward_index(index)),
                                        name=tensor.name + "_unpack")
                io_replace_dict[tensor] = unpack_output
        else:
            raise RuntimeError("Unsupported Operation")

    return io_replace_dict

def auto_pack(ginfo, cfg):
    """Pack tensors (transform layout) to better utilize vectorization and increase memory locality.
    This is for cpu only.

    Parameters
    ----------
    ginfo: GlobalInfo
        The global static analysis result
    cfg: ConfigEntity
        The configuration of current autotvm template
    """

    # only do pack for cpu
    if "cpu" not in _target.current_target().keys:
        return ginfo.bufs, False

    node_dict = ginfo.node_dict

    # plan new layout
    linked_axes = detect_linked_axes(node_dict)
    layout_dict, need_rewrite = plan_layout(node_dict, ginfo.bufs, linked_axes, cfg)

    if not need_rewrite:
        return ginfo.bufs, False

    # rewrite compute declaration
    io_replace_dict = rewrite_compute(node_dict, ginfo.bufs, layout_dict)

    return [io_replace_dict.get(x, x) for x in ginfo.bufs], True
