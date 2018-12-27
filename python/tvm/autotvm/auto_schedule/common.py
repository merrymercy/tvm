
class AutoScheduleOptions(object):
    """schedule options for the auto-scheduler"""

    # public accessors
    NUM_THREADS = 16
    VEC_SIZE = 8
    TUNING_LEVEL = 0
    PARALLEL_THRESHOLD = 1024

    _current = None

    def __init__(self, **kwargs):
        keys = [x for x in dir(AutoScheduleOptions) if not x.startswith('_')]

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


def _get_axis_length(axis):
    """Get the length of an axis. Returns 1 if any error occurs"""
    try:
        return axis.dom.extent.value
    except AttributeError:
        return 1
