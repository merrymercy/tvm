"""Stage analysis: Analyze compute ops and access relations

The result of this analysis is a StageGraph of type dict[op -> StageNode],
which maps raw operations to the node in the StageGraph.
The StageGraph is a DAG with the same structure as the original computational DAG,
but with more annotation on computation and access type.
"""

from ... import ir_pass as _ir_pass, tensor as _tensor, expr as _expr
from ...contrib.util import reg_enum_class

from .common import get_axis_length


@reg_enum_class
class StageNodeType:
    """Stage type"""
    'PLACEHOLDER'          # input placeholder
    'DIRECT_COMPUTE'       # direct compute
    'SIMPLE_REDUCTION'     # simple reduction without reuse opportunity
    'COMPLEX_REDUCTION'    # complex reduction with reuse opportunity


@reg_enum_class
class StageEdgeType:
    """Buffer read type"""
    'ELEMWISE'              # element wise
    'BROADCAST'             # broadcast
    'OTHER'                 # other
    'WEAK_INLINEABLE'       # weak inlineable access (e.g. elemwise with const shift padding)


@reg_enum_class
class ComputeAtType:
    """Compute at type"""
    'COMPUTE_ROOT'       # compute_root
    'COMPUTE_TUNE'       # compute_at with tunable knobs
    'COMPUTE_FUSE'       # compute_fuse to
    'COMPUTE_INLINE'     # compute_inline


class StageEdge:
    """Read access relation of a buffer"""
    def __init__(self):
        self.access = []   # list of Halide Call args
        self.type = None   # StageEdgeType


class StageNode:
    """A Stage in the computation graph"""
    def __init__(self, op, shape, ct):
        self.op = op
        self.shape = shape
        self.type = None       # StageNodeType
        self.read_edges = {}   # dict (StageNode -> StageEdge)
        self.write_edges = {}  # dict (StageNode -> StageEdge)

        self.name = op.name + "_" + str(ct)

        self.compute_at_loc = None  # StageNode or None (root)
        self.compute_at_type = ComputeAtType.COMPUTE_ROOT

    def __repr__(self):
        return self.name


def _node_set_in(a, b):
    """check whether a is a subset of b"""
    for x in a:
        if x not in b:
            return False
    return True


def _shape_same_as(a, b):
    """check whether two shapes are the same"""
    for x, y in zip(a, b):
        # todo(lmzheng): support symbolic check
        if not int(x) == int(y):
            return False
    return True


def _gather_access(body):
    """Gather all halide array accesses for an _expression"""
    access = []

    def _gather(stmt):
        if isinstance(stmt, _expr.Call) and stmt.call_type == _expr.Call.Halide:
            access.append((stmt.func, stmt.args))

    _ir_pass.PostOrderVisit(body, _gather)
    return access


def _gather_vars(exprs):
    """gather all iter vars from an expr or a list of exprs"""
    rets = []

    def _gather(stmt):
        if isinstance(stmt, _expr.Var):
            rets.append(stmt)

    if isinstance(exprs, _expr._expr):
        _ir_pass.PostOrderVisit(exprs, _gather)
    else:
        for arg in exprs:
            _ir_pass.PostOrderVisit(arg, _gather)
    return rets


def _reduce_node_reusable_axes(node):
    """check whether a reduce stage has the opportunity for memory reuse.
      If is true, we will do tiling on it
      Rule: For an spatial iteration axis, if it disappears in one buffer read, then we
            can get reuse opportunity by tiling on it.
      """
    axes = [x.var for x in node.op.axis if get_axis_length(x) > 1]

    direct_read = []
    for dst_node, edge in node.read_edges.items():
        for pattern in edge.access:
            # filter direct read
            item = [t for t in pattern if isinstance(t, _expr.Var)]
            direct_read.append(item)

    reusable_axes = []
    for x in axes:
        for item in direct_read:
            if x not in item:
                reusable_axes.append(x)
                break

    return reusable_axes


def _remove_const_shift(args):
    """extract access pattern but remove const shift.
    e.g. access pattern (n, c, h-1, w-1) in padding will be regard as a special elemwise op"""
    rets = []
    has_const_shift = False
    for x in args:
        if isinstance(x, _expr.IntImm):
            continue

        if (isinstance(x, (_expr.Add, _expr.Sub)) and
                ((isinstance(x.a, _expr.IntImm) and isinstance(x.b, _expr.IntImm))
                    or (isinstance(x.a, _expr.Var) and isinstance(x.b, _expr.IntImm)))):
            rets.append(x.a if isinstance(x.a, _expr.Var) else x.b)
            has_const_shift = True
        else:
            rets.append(x)

    return rets, has_const_shift


def _analyze_read_edges(node_dict, n):
    """analyze read edges for a node"""
    op = n.op

    # gather access pattern
    for body in op.body:
        for dst, pattern in _gather_access(body):
            dst_node = node_dict[dst]
            if dst_node not in n.read_edges:
                n.read_edges[dst_node] = StageEdge()

            n.read_edges[dst_node].access.append(pattern)

    op_axes = list([x.var for x in op.axis])

    # classify (read) edge type
    for dst_node, edge in n.read_edges.items():
        same = True
        if len(edge.access) > 1:
            x = edge.access[0]
            for y in edge.access[1:]:
                if not (len(x) == len(y) and all([a.same_as(b) for (a, b) in zip(x, y)])):
                    same = False
                    break

        if not same:
            edge.type = StageEdgeType.OTHER
        else:
            acc_axes, has_const_shift = _remove_const_shift(list(edge.access[0]))
            if _node_set_in(acc_axes, op_axes):
                if _node_set_in(op_axes, acc_axes):
                    edge.type = StageEdgeType.ELEMWISE
                else:
                    edge.type = StageEdgeType.BROADCAST
            else:
                edge.type = StageEdgeType.OTHER

                # try a rough analysis to detect layout transform style elemwise access
                #  if np.prod(get_const_tuple(n.op.output(0).shape)) == \
                #         np.prod(get_const_tuple(dst_node.op.output(0).shape)):
                #     match = True
                #     analyzer = arith.Analyzer()
                #     for v, extent in zip(op_axes, op.output(0).shape):
                #         analyzer.update(v, arith.ConstIntBound(0, extent.value-1))
                #     for _expr, extent in zip(acc_axes, dst_node.op.output(0).shape):
                #         bound = analyzer.const_int_bound(_expr)
                #         if bound.min_value != 0 or bound.max_value != extent.value - 1:
                #             match = False
                #             break
                #     if match:
                #         edge.type = StageEdgeType.ELEMWISE

            if has_const_shift and edge.type in [StageEdgeType.ELEMWISE, StageEdgeType.BROADCAST]:
                edge.type = StageEdgeType.WEAK_INLINEABLE

        node_dict[dst_node.op].write_edges[n] = edge


def build_stage_graph(bufs):
    """ Given input/output buffers, analyze access relation and build StageGraph

    Parameters
    ----------
    bufs: List of Tensor
        The input and output tensor

    Returns
    -------
    node_dict: dict[op -> StageNode]
        The dict maps raw ops to nodes in the stage graph.
    """
    node_dict = {}   # dict (op -> Node)

    # gather nodes
    que = list(bufs)
    i = 0
    while i < len(que):
        x = que[i]
        i += 1
        if x in node_dict:
            continue

        # TODO(lmzheng): handle multiple output
        node_dict[x.op] = StageNode(x.op, x.shape, i)
        if isinstance(x.op, _tensor.PlaceholderOp):
            pass
        elif isinstance(x.op, _tensor.ComputeOp):
            que.extend(x.op.input_tensors)
        else:
            raise ValueError("Unsupported operator type: " + str(type(x.op)))

    # classify node type
    for op, n in node_dict.items():
        if isinstance(op, _tensor.PlaceholderOp):
            n.type = StageNodeType.PLACEHOLDER
        elif isinstance(op, _tensor.ComputeOp):
            _analyze_read_edges(node_dict, n)
            if len(op.reduce_axis) == 0:
                n.type = StageNodeType.DIRECT_COMPUTE
            else:
                if _reduce_node_reusable_axes(n):
                    n.type = StageNodeType.COMPLEX_REDUCTION
                else:
                    n.type = StageNodeType.SIMPLE_REDUCTION
        else:
            raise RuntimeError("Unsupported Operators: " + str(type(op)))

    return node_dict


def topo_order(node_dict):
    """Get the topo order of the StageGarph (a directed acyclic graph)

    Parameters
    ----------
    node_dict: dict[op -> StageNode]
        The parsed stage graph

    Returns
    -------
    topo_order: List of StageNode
        The topo order of nodes
    """
    degree = {x: len(x.read_edges) for x in node_dict.values()}

    que = list(filter(lambda x: degree[x] == 0, node_dict.values()))
    ret = []

    head = 0
    while head < len(que):
        x = que[head]
        head += 1

        ret.append(x)
        for dst in x.write_edges:
            degree[dst] -= 1
            if degree[dst] == 0:
                que.append(dst)

    return ret

def annotate_compute_location(node_dict, bufs):
    """Divide stages into groups and annotate the compute locations
    (root, inline, tunable, etc) for stages.
    This function will modify the attributes of nodes directly.

    Parameters
    ----------
    node_dict: dict[op -> StageNode]
        The parsed stage graph

    Returns
    -------
    root_to_master: dict[StageNode -> StageNode]
         Maps a root op to the master op in its group (the complex reduction op)
    """
    output_ops = [x.op for x in bufs if hasattr(x, "op")]

    def _compute_root(n):
        n.compute_at_type = ComputeAtType.COMPUTE_ROOT
        n.compute_at_loc = n

    already_complex = set()  # record group that already has a complex op

    # use simple greedy alg to fuse stages
    for n in reversed(topo_order(node_dict)):
        if n.op in output_ops:
            _compute_root(n)
            continue

        if n.type in [StageNodeType.COMPLEX_REDUCTION, StageNodeType.SIMPLE_REDUCTION]:
            if len(n.write_edges) > 1:
                _compute_root(n)
            else:
                dst, edge = list(n.write_edges.items())[0]
                if (edge.type == StageEdgeType.ELEMWISE and dst.compute_at_loc is not None
                        and dst.compute_at_loc not in already_complex):
                    if dst.compute_at_type == ComputeAtType.COMPUTE_TUNE:
                        dst.compute_at_type = ComputeAtType.COMPUTE_ROOT
                        dst.compute_at_loc = dst
                    n.compute_at_type = ComputeAtType.COMPUTE_FUSE
                    n.compute_at_loc = dst.compute_at_loc
                    already_complex.add(dst.compute_at_loc)
                else:
                    _compute_root(n)
        elif n.type == StageNodeType.DIRECT_COMPUTE:
            if all([x.type in [StageEdgeType.ELEMWISE, StageEdgeType.BROADCAST,]
                    for x in n.read_edges.values()]):

                n.compute_at_type = ComputeAtType.COMPUTE_INLINE

                for dst, edge in n.write_edges.items():
                    if edge.type == StageEdgeType.ELEMWISE and _shape_same_as(dst.shape, n.shape):
                        n.compute_at_loc = dst.compute_at_loc  # path compression, this is for fused op to
                        break                                  # find the final output
            else:
                if len(n.write_edges) == 1:
                    n.compute_at_type = ComputeAtType.COMPUTE_TUNE
                    n.compute_at_loc = list(n.write_edges.keys())[0]
                else:
                    _compute_root(n)
        else:
            _compute_root(n)

    # build dict that maps a root op to the master op in its group (the complex reduction op)
    root_to_master = dict()
    for n in node_dict.values():
        if n.compute_at_type == ComputeAtType.COMPUTE_FUSE:
            assert n.compute_at_loc not in root_to_master,\
                "Fuse multiple complex compute nodes into a single one"
            assert n.compute_at_loc.compute_at_type == ComputeAtType.COMPUTE_ROOT
            root_to_master[n.compute_at_loc] = n

    return root_to_master


def print_stage_graph(node_dict):
    """Print stage graph for debug usage

    Parameters
    ----------
    node_dict: dict[op -> StageNode]
        The parsed stage graph
    """
    for op, node in node_dict.items():
        print("============")
        print(op, StageNodeType.to_str(node_dict[op].type), op.output(0).shape)
        for dst, edge in node.read_edges.items():
            print("read:", dst.op, StageEdgeType.to_str(edge.type), edge.access)
        print("compute_at:", ComputeAtType.to_str(node.compute_at_type), node.compute_at_loc)
    print("============")
