# pylint: disable=invalid-name
"""Extract feature of iter vars

There are two types of feature
1) Itervar feature
   This feature is extracted based on loop variables.
   Different loop structures will result in different shapes of feature
2) Curve sample feature (relation feature)
   This feature is extracted by sampling relation curve.
   This feature is invariant of loop structure.
3) Simplified AST
   Extract feature of simplified AST for TreeRNN
"""

import struct
import numpy as np

from tvm import schedule, ir_pass, build_module, get_global_func, target as _target

def ana_lower(sch, args,
              binds=None,
              simple_mode=True):
    """Do lower while keeping all axes in IR
    i.e. Do not eliminate loop with extent of 1, do not vectorize, unroll or inject virtual threads
    """
    binds, _ = build_module.get_binds(args, binds)
    sch = sch.normalize()
    # Phase 0
    bounds = schedule.InferBound(sch)
    stmt = schedule.ScheduleOps(sch, bounds, True)
    stmt = ir_pass.StorageFlatten(stmt, binds, 64)
    stmt = ir_pass.CanonicalSimplify(stmt)
    assert simple_mode
    return stmt

try:
    _get_buffer_curve_sample_flatten = get_global_func(
        "autotvm.feature.GetCurveSampleFeatureFlatten")
    _get_itervar_feature = get_global_func("autotvm.feature.GetItervarFeature")
    _get_itervar_feature_flatten = get_global_func("autotvm.feature.GetItervarFeatureFlatten")
    _get_simplified_ast = get_global_func("autotvm.feature.GetSimplifiedAST")
except ValueError as e:
    def raise_error(*args, **kwargs):  # pylint: disable=unused-argument
        raise RuntimeError("Cannot load autotvm c++ API")
    _get_buffer_curve_sample_flatten = _get_itervar_feature = _get_itervar_feature_flatten = \
        raise_error

def get_itervar_feature(sch, args, take_log=False):
    """get features of iter vars

    Parameters
    ----------
    sch: tvm.schedule.Schedule
    args: Array of tvm.tensor.Tensor
        the buffer args for lower
    take_log: bool
        whether take log of numerical statics

    Returns
    -------
    features of every axis in the IR, see doc/features.md for detail
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_itervar_feature(stmt, take_log)

    # convert tvm node to python type
    ret = []
    for row in feas:
        tmp = []
        tmp.append([row[0][0].value, row[0][1]])
        for item in row[1:]:
            tmp.append([item[0].value] + [x.value for x in item[1:]])
        ret.append(tmp)
    return ret

def flatten_itervar_feature(fea):
    """flatten features into one-dimensional feature vectors

    Parameters
    ----------
    fea: list
        return value of get_itervar_feature

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    flatten = []
    for axis in fea:
        for pair in axis[1:]:
            flatten.append(pair[1:])
    return np.concatenate(flatten)

def get_itervar_feature_flatten(sch, args, take_log=True):
    """get flatten features of iter vars
    this is equivalent to get_itervar_feature + flatten_itervar_feature, but much faster.

    Parameters
    ----------
    sch: tvm.schedule.Schedule
    args: Array of tvm.tensor.Tensor
        the buffer args for lower
    take_log: bool
        whether take log of numerical statics

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_itervar_feature_flatten(stmt, take_log)
    feas = struct.unpack('%df' % (len(feas)//4), feas)
    return feas

def get_flatten_name(fea):
    """ Get names of feature after flatten.

    Parameters
    ----------
    fea: list or str
        return value of get_itervar_feature or a line of logfile

    Returns
    -------
    feature_names: Array of str
    """

    feature_name = {
        "_attr_": ["length", "nest_level", "topdown", "bottomup"] +
                  ["ann_%d" % i for i in range(20)],
        "_arith_": ["add", "mul", "div"],
        "buf_touch": ["stride", "mod", "count", "reuse", "T_count", "T_reuse"],
    }

    if isinstance(fea, str):
        from .record import decode
        # flatten line to feature
        line = fea
        inp, _ = decode(line)
        with inp.target:
            s, args = inp.task.instantiate(inp.config)
        fea = get_itervar_feature(s, args)

    names = []
    ct = 0
    for row in fea:
        var_name = str(row[0][1])
        for pair in row[1:]:
            key = pair[0]
            if key in feature_name:
                name_list = feature_name[key]
            else:
                name_list = feature_name["buf_touch"]

            for i in range(len((pair[1:]))):
                names.append(".".join(["f%d" % ct, var_name, key, name_list[i]]))
                ct += 1
    return names


def get_buffer_curve_sample_flatten(sch, args, sample_n=30):
    """
    Get flatten curve sample feature (relation feature)

    Parameters
    ----------
    sch: tvm.schedule.Schedule
    args: Array of tvm.tensor.Tensor
        the buffer args for lower
    sample_n: int
        number of sample points along one dimension

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_buffer_curve_sample_flatten(stmt, sample_n, False)
    feas = struct.unpack('%df' % (len(feas)//4), feas)
    return feas


_vocabulary = {}
MAX_NUM_CHILDREN = 20
MAX_DIM_ADDITIONAL = 24   # 12 + ...

def clean_vocabulary():
    """clear token collecting vocabulary"""
    global _vocabulary
    _vocabulary = {}

def get_simplified_ast(sch, args, take_log=True, keep_name=False, add_stats=True):
    """get simplified AST feature for tree RNN

    Parameters
    ----------
    sch: tvm.schedule.Schedule
    args: Array of tvm.tensor.Tensor
        The buffer args for lower
    take_log: bool
        Whether take log of numerical statistics
    keep_name: bool
        Whether keep str name in return value
        if it is true, then returned emb_idx is Array of str
        if it is false, then returned emb_idx is Array of int
    add_stats: bool
        Whether add add statistics feature feature

    Returns
    -------
    children: np.ndarray
        Two dimensional array to store children info
        Each nodes has a row, each row is (n_children, child_1, child_2, child_3, xxx)
    emb_idx: np.ndarray of int or str
        If keep_name is true,  n_tree x str    (name)
        If keep_name is false, n_tree x int32  (unique id)
    additional_feature: np.ndarray
        Each nodes has a row
        Each row stores additional feature for this node (e.g. length, one-hot annotation)
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_simplified_ast(stmt, add_stats)

    n_tree, offset_child, offset_names, offset_addfea, = \
        struct.unpack('4i', feas[:4 * 4])

    # unpack byte stream
    child_nums = struct.unpack("%di" % n_tree, feas[offset_child:offset_child + 4 * n_tree])
    children = []
    offset_child += 4 * n_tree
    ct = 0
    for i in range(n_tree):
        num = child_nums[i]
        children.append(struct.unpack("%di" % num,
                                      feas[offset_child + ct * 4: offset_child + (ct + num) * 4]))
        ct += num

    name_nums = struct.unpack("%di" % n_tree, feas[offset_names:offset_names + 4 * n_tree])
    names = []
    offset_names += 4 * n_tree
    ct = 0
    for i in range(n_tree):
        num = name_nums[i]
        names.append(struct.unpack("%ds" % num,
                                   feas[offset_names + ct: offset_names + ct + num])[0].decode())
        ct += num

    addfea_nums = struct.unpack("%di" % n_tree, feas[offset_addfea:offset_addfea + 4 * n_tree])
    add_feas = []
    offset_addfea += 4 * n_tree
    ct = 0
    for i in range(n_tree):
        num = addfea_nums[i]
        add_feas.append(struct.unpack("%df" % num,
                                      feas[offset_addfea + ct * 4: offset_addfea + (ct + num) * 4]))
        ct += num

    # copy children relation to numpy array (with padding)
    children_np = np.empty((n_tree, MAX_NUM_CHILDREN + 1), dtype=np.int16)
    for i, chi in enumerate(children):
        n_chi = len(chi)
        children_np[i, 0] = n_chi
        children_np[i, 1:1 + n_chi] = chi

    # transform string name to integer index
    emb_idx_np = np.empty((n_tree,), dtype=object if keep_name else np.int16)
    for i, name in enumerate(names):
        if name not in _vocabulary:
            _vocabulary[name] = len(_vocabulary)
        emb_idx_np[i] = (name if keep_name else _vocabulary[name])

    # copy additional feature to numpy array (with padding)
    add_feas_np = np.zeros((n_tree, MAX_DIM_ADDITIONAL), dtype=np.float32)
    for i, fea in enumerate(add_feas):
        add_feas_np[i, :addfea_nums[i]] = fea

    if take_log:
        log_index = add_feas_np > 0
        add_feas_np[log_index] = np.log2(add_feas_np[log_index] + 1.0)

    return children_np, emb_idx_np, add_feas_np
