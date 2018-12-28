from .... import target as _target

@_target.generic_func
def schedule_complex_reduce(s, cfg, node):
    raise NotImplementedError()

@_target.generic_func
def schedule_simple_reduce(s, cfg, node):
    raise NotImplementedError()

@_target.generic_func
def schedule_other_root(s, cfg, node):
    raise NotImplementedError()
