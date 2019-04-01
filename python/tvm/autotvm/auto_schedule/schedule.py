"""The auto_scheduler"""

from ... import schedule
from ..task.task import get_config
from ..task.space import ConfigSpace
from .common import AutoScheduleOptions as opts
from .stage_analysis import build_stage_graph, annotate_compute_location, \
    StageNodeType, ComputeAtType, print_stage_graph
from .backend.generic import schedule_complex_reduce, schedule_simple_reduce, \
    schedule_other_root, schedule_tune_direct_compute
from .auto_packing import auto_pack

class GlobalInfo:
    """Global cached analysis results"""
    def __init__(self):
        self.node_dict = None       # dict[op -> StageNode], the stage graph
        self.root_to_master = None  # dict[StageNode -> StageNode], auxiliary info for compute location
        self.bufs = []              # list[Tensor], arguments bufs


def call_schedule_func(ginfo, cfg):
    """call corresponding schedule func

    Parameters
    ----------
    ginfo: GlobalInfo
        The cached global info
    cfg: Union[ConfigEntity, ConfigSpace]
        The current config
    """
    node_dict = ginfo.node_dict
    s = schedule.create_schedule([x.op for x in node_dict.values() if len(x.write_edges) == 0])

    root_to_master = ginfo.root_to_master

    # call schedule for nodes that compute at root
    for node in node_dict.values():
        if node.compute_at_type == ComputeAtType.COMPUTE_INLINE:
            s[node.op].compute_inline()
        elif node.compute_at_type == ComputeAtType.COMPUTE_ROOT:
            master = root_to_master.get(node, node)

            if master.type == StageNodeType.COMPLEX_REDUCTION:
                schedule_complex_reduce(s, cfg, master)
            elif master.type == StageNodeType.SIMPLE_REDUCTION:
                schedule_simple_reduce(s, cfg, master)
            elif master.type == StageNodeType.PLACEHOLDER:
                pass
            else:
                schedule_other_root(s, cfg, master)

    # call schedule for nodes with tunable compute location
    for node in node_dict.values():
        if node.compute_at_type == ComputeAtType.COMPUTE_TUNE:
            schedule_tune_direct_compute(s, cfg, node)

    return s

def prune_space(cfg):
    """Prune the tuning space.

    Parameters
    ----------
    cfg: ConfigSpace
        The current config space
    """
    # remove higher tuning level knobs
    if isinstance(cfg, ConfigSpace):
        keys = list(cfg.space_map)
        for k in keys:
            level = int(k[1])
            if level > opts.TUNING_LEVEL:
                del cfg.space_map[k]
                del cfg._entity_map[k]

# cache for analysis results
GINFO_CACHE = dict()

def static_analysis(bufs):
    """Do static analysis of stages

    Parameters
    ----------
    bufs: List of Tensor
        The input/output tensors

    Returns
    -------
    ginfo: GlobalInfo
        The static analysis result
    """
    # stage analysis
    node_dict = build_stage_graph(bufs)

    # annotate compute location
    root_to_master = annotate_compute_location(node_dict, bufs)

    # cache analysis result
    ginfo = GlobalInfo()
    ginfo.node_dict = node_dict
    ginfo.bufs = bufs
    ginfo.root_to_master = root_to_master

    return ginfo

def create_schedule(bufs):
    """Create schedule for bufs.

    Parameters
    ----------
    bufs: List of tensors
        The input/output tensors

    Returns
    -------
    sch: Schedule
        The created schedule
    bufs: List of tensors
        The optimized input/output tensors.
        The tensors in this list exactly matches the arguments in shape
        and semantics but with different address due to some compute
        deceleration rewrite.

    Notes
    -----
    1. This function is a tunable autotvm template.
       You can call this function under different configurations
    2. Some optimization pass will rewrite compute deceleration,
       so you should use the returned bufs instead of original tensors.
    """
    cfg = get_config()

    # analyze stages
    ginfo = static_analysis(bufs)

    #print_stage_graph(ginfo.node_dict)

    # compute rewrite
    if opts.AUTO_PACK:
        bufs, modified = auto_pack(ginfo, cfg)
        if modified:
            ginfo = static_analysis(bufs)
        # return _create_schedule([bufs[-1].op]), bufs

    # call schedule template
    sch = call_schedule_func(ginfo, cfg)

    # prune space
    prune_space(cfg)

    return sch, bufs
