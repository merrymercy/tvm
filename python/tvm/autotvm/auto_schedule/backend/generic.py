"""The generic schedule interface.
These function will be overloaded by different backends"""

from .... import target as _target

@_target.generic_func
def schedule_tune_direct_compute(s, cfg, node):
    """Schedule tunable direct compute op (e.g. elemwise with tunable location)"""
    raise NotImplementedError()

@_target.generic_func
def schedule_other_root(s, cfg, node):
    """Schedule other compute type (e.g. elemwise, broadcast)"""
    raise NotImplementedError()

@_target.generic_func
def schedule_simple_reduce(s, cfg, node):
    """Schedule simple reduction op (e.g. softmax, min, max)"""
    raise NotImplementedError()

@_target.generic_func
def schedule_complex_reduce(s, cfg, node):
    """Schedule complex reduction op (e.g. matmul, conv)"""
    raise NotImplementedError()
