import time
import logging

import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm import MeasureInput, MeasureResult
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel
from tvm.autotvm.tuner.treernn_cost_model import TreeRNNCostModel
from test_autotvm_common import get_sample_task, get_sample_records


def test_fit_xgb():
    task, target = get_sample_task()
    records = get_sample_records(n=500)

    xgb_params = {
        'max_depth': 3,
        'gamma': 0.0001,
        'min_child_weight': 1,

        'subsample': 1.0,

        'eta': 0.3,
        'lambda': 1.00,
        'alpha': 0,

        'objective': 'reg:linear',
    }

    base_model = XGBoostCostModel(task, feature_type='itervar', xgb_params=xgb_params)
    base_model.fit_log(records, plan_size=32)

    upper_model = XGBoostCostModel(task, feature_type='itervar', xgb_params=xgb_params)
    upper_model.load_base_model(base_model)

    xs = np.arange(10)
    ys = np.arange(10)

    upper_model.fit(xs, ys, plan_size=32)

def test_fit_treernn():
    task, target = get_sample_task()
    records = get_sample_records(n=500)

    rnn_params = {
	'ctx': 'gpu',
	'cell_type': 'gru',
	'voc_size': 128,
	'emb_dim': 128,
	'rnn_hidden_size': 128,
	'decoder_hidden_size': [128],
	'max_n_children': 20,
	'num_mem_slots': 0,

	'train_batch_size': 128,
	'eval_batch_size': 1024,

	'train_eval_every': 50,
	'train_early_stopping': 100,

	'max_batch': 1000,
	'learning_rate': 7e-4,
	'wd': 1e-4,
	'clip_gradient': 5.0,

	'loss_type': 'reg',
    }

    xs = np.arange(100)
    ys = np.arange(100)

    base_model = TreeRNNCostModel(task, feature_type='itervar', rnn_params=rnn_params)
    base_model.fit(xs, ys, plan_size=32)

def test_tuner():
    task, target = get_sample_task()
    records = get_sample_records(n=100)

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.load_history(records)

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger('autotvm')
    logger.setLevel(logging.DEBUG)

    test_fit_treernn()
    #test_fit_xgb()
    #test_tuner()

