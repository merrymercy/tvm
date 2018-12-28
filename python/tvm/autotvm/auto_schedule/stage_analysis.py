"""Stage analysis"""

from ... import ir_pass, tensor, expr
from ...contrib.util import reg_enum_class

from .common import _get_axis_length


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
        self.type = None


class StageNode:
    """A Stage in the computation graph"""
    def __init__(self, op, shape, ct):
        self.op = op
        self.shape = shape
        self.type = None
        self.read_edges = {}   # dict (StageNode -> StageEdge)
        self.write_edges = {}  # dict (StageNode -> StageEdge)

        self.name = op.name + "_" + str(ct)

        self.compute_at_loc = None  # StageNode or None (root)
        self.compute_at_type = ComputeAtType.COMPUTE_ROOT


def _node_set_in(a, b):
    """check whether a is a subset of b"""
    for x in a:
        if not any([x.same_as(y) for y in b]):
            return False
    return True

def _gather_access(body):
    """Gather all accesses for an expression"""
    access = []

    def _gather(stmt):
        if isinstance(stmt, expr.Call) and stmt.call_type == expr.Call.Halide:
            access.append((stmt.func, stmt.args))

    ir_pass.PostOrderVisit(body, _gather)
    return access


def _gather_vars(args):
    """gather all iter vars"""
    rets = []

    def _gather(stmt):
        if isinstance(stmt, expr.Var):
            rets.append(stmt)

    for arg in args:
        ir_pass.PostOrderVisit(arg, _gather)
    return rets


def _reduce_node_has_reuse(node):
    """check whether a reduce stage has the opportunity for memory reuse.
      If is true, we will do tiling on it"""
    spatial_axes = set([x.var for x in node.op.axis if _get_axis_length(x) > 1])
    n_loss = 0
    for dst_node, edge in node.read_edges.items():
        for pattern in edge.access:
            itervars = _gather_vars(pattern)
            if not _node_set_in(spatial_axes, itervars):
                n_loss += 1
                break

    if n_loss == len(node.read_edges):
        return True
    else:
        return False


def _remove_const_shift(args):
    """extract access pattern but remove const shift.
    e.g. access pattern (n, c, h-1, w-1) in padding will be regard as a special elemwise op"""
    rets = []
    has_const_shift = False
    for x in args:
        if (isinstance(x, (expr.Add, expr.Sub)) and
                ((isinstance(x.a, expr.IntImm) and isinstance(x.b, expr.IntImm))
                    or (isinstance(x.a, expr.Var) and isinstance(x.b, expr.IntImm)))):
            rets.append(x.a if isinstance(x.a, expr.Var) else x.b)
            has_const_shift = True
        else:
            rets.append(x)

    return rets, has_const_shift


def insert_read_edges(node_dict, n):
    """insert read edges for a node"""
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

            if has_const_shift and edge.type in [StageEdgeType.ELEMWISE, StageEdgeType.BROADCAST]:
                edge.type = StageEdgeType.WEAK_INLINEABLE

        node_dict[dst_node.op].write_edges[n] = edge


def build_stage_graph(bufs):
    """ Given bufs, output StageGraph """
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
        if isinstance(x.op, tensor.PlaceholderOp):
            pass
        elif isinstance(x.op, tensor.ComputeOp):
            que.extend(x.op.input_tensors)
        else:
            raise ValueError("Unsupported operator type: " + str(type(x.op)))

    # identify node type
    for op, n in node_dict.items():
        if isinstance(op, tensor.PlaceholderOp):
            n.type = StageNodeType.PLACEHOLDER
        elif isinstance(op, tensor.ComputeOp):
            insert_read_edges(node_dict, n)
            if len(op.reduce_axis) == 0:
                n.type = StageNodeType.DIRECT_COMPUTE
            else:
                if _reduce_node_has_reuse(n):
                    n.type = StageNodeType.COMPLEX_REDUCTION
                else:
                    n.type = StageNodeType.SIMPLE_REDUCTION
        else:
            raise RuntimeError("Unsupported Operators: " + str(type(op)))

    return node_dict


def _topo_sort(node_dict):
    """topo sort nodes"""
    degree = {x: len(x.write_edges) for x in node_dict.values()}

    que = list(filter(lambda x: degree[x] == 0, node_dict.values()))
    ret = []

    head = 0
    while head < len(que):
        x = que[head]
        head += 1

        ret.append(x)
        for dst in x.read_edges:
            degree[dst] -= 1
            if degree[dst] == 0:
                que.append(dst)

    assert len(ret) == len(node_dict)
    return ret


def _shape_same_as(a, b):
    """check whether two shares are the same"""
    for x, y in zip(a, b):
        # todo(lmzheng): support symbolic check
        if not int(x) == int(y):
            return False
    return True


def annotate_compute_location(node_dict, bufs):
    """Divide operators into groups and annotate the compute locations for stages"""
    output_ops = [x.op for x in bufs if hasattr(x, "op")]

    def _compute_root(n):
        n.compute_at_type = ComputeAtType.COMPUTE_ROOT
        n.compute_at_loc = n

    already_complex = set()  # record group that already has a complex op

    # simple greedy fusion
    for n in _topo_sort(node_dict):
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
                    n.compute_at_type = ComputeAtType.COMPUTE_FUSE
                    n.compute_at_loc = dst.compute_at_loc
                    already_complex.add(dst.compute_at_loc)
                else:
                    _compute_root(n)
        elif n.type == StageNodeType.DIRECT_COMPUTE:
            if all([x.type in [StageEdgeType.ELEMWISE, StageEdgeType.BROADCAST,
                               StageEdgeType.WEAK_INLINEABLE]
                    for x in n.read_edges.values()]):

                n.compute_at_type = ComputeAtType.COMPUTE_INLINE

                for dst, edge in n.write_edges.items():
                    if edge.type == StageEdgeType.ELEMWISE and _shape_same_as(dst.shape, n.shape):
                        n.compute_at_loc = dst.compute_at_loc
                        break
            else:
                _compute_root(n)
        else:
            _compute_root(n)

    root_to_master = dict()
    for n in node_dict.values():
        if n.compute_at_type == ComputeAtType.COMPUTE_FUSE:
            assert n.compute_at_loc not in root_to_master
            root_to_master[n.compute_at_loc] = n

    return root_to_master


def print_stage_graph(node_dict):
    """Print stage graph for debug usage"""
    for op, node in node_dict.items():
        print("============")
        print(op, StageNodeType.to_str(node_dict[op].type))
        for dst, edge in node.read_edges.items():
            print("read:", dst.op, StageEdgeType.to_str(edge.type), edge.access)
        print("compute_at:", ComputeAtType.to_str(node.compute_at_type), bool(node.compute_at_loc))
