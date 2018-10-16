"""Tuner that uses xgboost as cost model"""

from .model_based_tuner import ModelBasedTuner, ModelOptimizer
from .treernn_cost_model import TreeRNNCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer


class TreeGRUTuner(ModelBasedTuner):
    """Tuner that uses xgboost as cost model

    Parameters
    ----------
    task: Task
        The tuning task
    plan_size: int
        The size of a plan. After `plan_size` trials, the tuner will refit a new cost model
        and do planing for the next `plan_size` trials.
    feature_type: str, optional

    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
    num_threads: int, optional
        The number of threads.  optimizer: str or ModelOptimizer, optional
        If is 'sa', use a default simulated annealing optimizer.
        Otherwise it should be a ModelOptimizer object.
    diversity_filter_ratio: int or float, optional
        If is not None, the tuner will first select
        top-(plan_size * diversity_filter_ratio) candidates according to the cost model
        and then pick batch_size of them according to the diversity metric.
    log_interval: int, optional
        The verbose level.
        If is 0, output nothing.
        Otherwise, output debug information every `verbose` iterations.
    rnn_params: Dict, optional
        The parameters for xgboost model. This will override `loss_type`
    """
    def __init__(self, task, plan_size=64,
                 feature_type='itervar', loss_type='rank', num_threads=None,
                 optimizer='sa', diversity_filter_ratio=None, log_interval=50, rnn_params=None):

        if rnn_params is None:
            if loss_type == 'reg':
                rnn_params = {
                    # rnn setting
                    'cell_type': 'gru',
                    'voc_size': 128,
                    'emb_dim': 128,
                    'rnn_hidden_size': 128,
                    'decoder_hidden_size': [128],
                    'max_n_children': 20,
                    'num_mem_slots': 0,

                    # training setting
                    'loss_type': 'reg',
                    'learning_rate': 7e-4,
                    'wd': 1e-4,
                    'clip_gradient': 5.0,

                    'ctx': 'gpu',
                    'max_batch': 500,
                    'train_batch_size': 128,
                    'train_eval_every': 10,
                    'train_early_stopping': 100,
                }
            elif loss_type == 'rank':
                rnn_params = {
                    # rnn setting
                    'cell_type': 'gru',
                    'voc_size': 128,
                    'emb_dim': 128,
                    'rnn_hidden_size': 128,
                    'decoder_hidden_size': [128],
                    'max_n_children': 20,
                    'num_mem_slots': 0,

                    # training setting
                    'loss_type': 'rank',
                    'learning_rate': 7e-4,
                    'wd': 1e-4,
                    'clip_gradient': 5.0,

                    'ctx': 'gpu',
                    'max_batch': 500,
                    'train_batch_size': 128,
                    'train_eval_every': 10,
                    'train_early_stopping': 100,
                }
            else:
                raise RuntimeError("Invalid loss type: " + loss_type)

        cost_model = TreeRNNCostModel(task,
                                      feature_type=feature_type,
                                      rnn_params=rnn_params,
                                      num_threads=num_threads,
                                      log_interval=log_interval // 2)

        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."

        super(TreeGRUTuner, self).__init__(task, cost_model, optimizer,
                                           plan_size, diversity_filter_ratio)

    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(TreeGRUTuner, self).tune(*args, **kwargs)

        # manually close pool to avoid multiprocessing issues
        self.cost_model._close_pool()
