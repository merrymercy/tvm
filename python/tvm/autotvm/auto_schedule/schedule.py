from ... import schedule
from ..task.task import get_config
from ..task.space import get_factors, ConfigEntity, ConfigSpace, FallbackConfigEntity
from .stage_analysis import build_stage_graph, annotate_compute_location, \
    print_stage_graph, StageNodeType, ComputeAtType
from .common import AutoScheduleOptions as opts
from .backend.generic import schedule_complex_reduce, schedule_simple_reduce, \
    schedule_other_root, schedule_tune_direct_compute

class GlobalInfo:
    """global cached analysis results"""
    def __init__(self):
        self.node_dict = None       # dict[op -> StageNode]
        self.root_to_master = None  # dict[StageNode -> StageNode]
        self.bufs = []              # arguments bufs


def call_schedule_func(ginfo, cfg):
    """call corresponding schedule func"""
    node_dict = ginfo.node_dict
    s = schedule.create_schedule([x.op for x in node_dict.values() if len(x.write_edges) == 0])

    root_to_master = ginfo.root_to_master

    # call real schedule on roots
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

    # tunable compute location
    for node in node_dict.values():
        if node.compute_at_type == ComputeAtType.COMPUTE_TUNE:
            schedule_tune_direct_compute(s, cfg, node)

    return s

def prune_space(cfg):
    """Prune the tuning space"""

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

def create_schedule(bufs):
    cfg = get_config()

    # stage analysis
    node_dict = build_stage_graph(bufs)

    # annotate compute location
    root_to_master = annotate_compute_location(node_dict, bufs)

    # DEBUG
    # print_stage_graph(node_dict)

    ginfo = GlobalInfo()
    ginfo.node_dict = node_dict
    ginfo.bufs = bufs
    ginfo.root_to_master = root_to_master

    # autotvm part
    s = call_schedule_func(ginfo, cfg)

    # prune space
    prune_space(cfg)

    return s
