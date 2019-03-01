from .... import target as _target

@_target.generic_func
def schedule_complex_reduce(s, cfg, node):
    """schedule complex reduction (e.g. matmul, conv)"""
    raise NotImplementedError()

@_target.generic_func
def schedule_simple_reduce(s, cfg, node):
    """schedule simple reduction (e.g. softmax, min, max)"""
    raise NotImplementedError()

@_target.generic_func
def schedule_other_root(s, cfg, node):
    """schedule other compute type (e.g. elemwise, broadcast)"""
    raise NotImplementedError()

@_target.generic_func
def schedule_tune_direct_compute(s, cfg, node):
    """schedule tunable direct compute node (e.g. elemwise with tunable location)"""
    raise NotImplementedError()
