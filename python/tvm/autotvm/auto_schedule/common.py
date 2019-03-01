"""Common utility for the auto-scheduler"""

from ..task.space import FallbackConfigEntity

class AutoScheduleOptions(object):
    """ Schedule options for the auto-scheduler, which includes
    - hardware parameters: like number of threads, vector size
    - tuning level: the higher, more knobs will be tuned
    """

    # public accessors
    TUNING_LEVEL = 1  # 0 -> tune nothing, 3 -> tune all knobs. 1 is the typical value

    NUM_THREADS = 16   # number of threads for CPU
    TILE_SIZE = 8      # default tile size for CPU
    VEC_SIZE = 8       # default vector size for CPU
    MAX_UNROLL = 32    # max unroll factor for CPU
    CACHE_SIZE = 8192  # L1 cache size for CPU

    MAX_GPU_THREADS = 1024  # maximum number of threads for GPU

    _current = None

    def __init__(self, **kwargs):
        keys = [x for x in dir(AutoScheduleOptions) if not x.startswith('_')]
        kwargs = {k.upper(): v for k, v in kwargs.items()}

        for k, _ in kwargs.items():
            if k not in keys:
                raise ValueError("invalid argument %s, candidates are %s" % (k, keys))
        self._old_scope = None

        self.attr = {k: getattr(AutoScheduleOptions, k) for k in keys}
        self.attr.update(kwargs)

    def __enter__(self):
        self._old_scope = AutoScheduleOptions._current
        AutoScheduleOptions.set_current(self)
        return self

    def __exit__(self, ptype, value, trace):
        AutoScheduleOptions.set_current(self._old_scope)

    @staticmethod
    def set_current(scope):
        """set the current scope and copy its attributes to public accessors"""
        AutoScheduleOptions._current = scope
        for k, v in scope.attr.items():
            setattr(AutoScheduleOptions, k, v)

AutoScheduleOptions.set_current(AutoScheduleOptions())

def tuning_level(cfg):
    """Return the current tuning level"""
    if isinstance(cfg, FallbackConfigEntity):
        return 0
    else:
        return AutoScheduleOptions.TUNING_LEVEL

def _get_axis_length(axis):
    """Get the length of an axis. Returns 1 if any error occurs"""
    try:
        return axis.dom.extent.value
    except AttributeError:
        try:
            return axis.attached_length
        except AttributeError:
            return 1
