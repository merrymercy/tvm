# pylint: disable=invalid-name
"""TreeGRU as cost model"""

import multiprocessing
import logging
import time

import numpy as np

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

from ..feature import get_simplified_ast
from .fold.rnn import ChildSumGRUCell, ChildSumLSTMCell
from .metric import average_recall
from .model_based_tuner import CostModel, FeatureCache

logger = logging.getLogger('autotvm')


class TreeRNNBlock(nn.Block):
    """Gluon Block of TreeRNN for Cost Prediction"""
    def __init__(self, mode, voc_size, emb_dim, rnn_hidden_size,
                 decoder_hidden_size, max_n_children=12, num_mem_slots=0,
                 act='sigmoid'):
        super(TreeRNNBlock, self).__init__()

        self.num_mem_slots = num_mem_slots
        self.rnn_hidden_size = rnn_hidden_size

        with self.name_scope():
            if emb_dim == 0:  # do not use embedding, only use stats info
                self.embed = None
                self.input_encode = nn.Dense(rnn_hidden_size, activation=act)
            else:
                self.embed = nn.Embedding(voc_size, emb_dim)
                self.embed.hybridize()

            if mode == 'lstm':
                cell = ChildSumLSTMCell
                self.cell_class = cell
            elif mode == 'gru':
                cell = ChildSumGRUCell
                self.cell_class = cell
            else:
                raise ValueError("invalid mode " + mode)

            self.childsumrnn = cell(rnn_hidden_size, 0, input_size=emb_dim)

            if self.num_mem_slots != 0:
                self.to_slots = nn.Dense(num_mem_slots, in_units=rnn_hidden_size,
                                         use_bias=False, activation=act)
                self.to_slots.hybridize()
                self.decoder_0 = nn.Dense(decoder_hidden_size[0],
                                          in_units=self.num_mem_slots * rnn_hidden_size,
                                          activation=act)
            else:
                self.to_slots = None
                self.decoder_0 = nn.Dense(decoder_hidden_size[0],
                                          in_units=rnn_hidden_size,
                                          activation=act)

            self.decoder_1 = nn.Dense(1, in_units=decoder_hidden_size[0])

            self.decoder_0.hybridize()
            self.decoder_1.hybridize()

        self.cells = {0: self.childsumrnn}
        for i in range(1, max_n_children):
            self.cells[i] = cell(rnn_hidden_size, i, input_size=emb_dim,
                                 params=self.childsumrnn.collect_params())

        for c in self.cells.values():
            c.hybridize()

    def forward(self, tree, emb_idx, additional):
        if self.embed:
            embeddings = self.embed(emb_idx)
            inputs = mx.nd.concat(additional, embeddings, dim=1)
        else:
            inputs = self.input_encode(additional)
        out = self.cell_class.encode(self.cells, inputs, tree)[1][0]
        out = self.decoder_0(out)
        out = self.decoder_1(out)
        return out

    def batch_forward(self, idx, children, emb_idxs, additionals):
        """forward for a batch of AST with the same tree structure"""
        if self.embed:
            embeddings = self.embed(emb_idxs)
            inputs = mx.nd.concat(additionals, embeddings, dim=-1)
        else:
            n_batch, n_tree = additionals.shape[:2]
            additionals = additionals.reshape((n_batch * n_tree, -1))
            inputs = self.input_encode(additionals)
            inputs = inputs.reshape((n_batch, n_tree, -1))
        mem_slots = [] if self.num_mem_slots != 0 else None
        out, _ = self.cell_class.batch_forward(
            idx, children, inputs, self.to_slots, self.num_mem_slots, mem_slots, self.cells)
        if self.num_mem_slots != 0:
            out = mx.nd.add_n(*mem_slots)
        out = self.decoder_0(out)
        out = self.decoder_1(out)
        return out

    def fold_encode(self, fold, tree, emb_idx, additional):
        embeddings = self.embed(emb_idx)
        inputs = mx.nd.concat(embeddings, additional, dim=1)
        out = self.cell_class.fold_encode(fold, self.cells, inputs, tree)[1][0]
        out = fold.record(0, self.decoder_0, out)
        out = fold.record(0, self.decoder_1, out)
        return out


class TreeRNNCostModel(CostModel):
    """TreeGRU as cost model

    Parameters
    ----------
    """
    def __init__(self, task, feature_type, rnn_params,
                 num_threads=None, log_interval=25, upper_model=None):
        super(TreeRNNCostModel, self).__init__()

        self.task = task
        self.target = task.target
        self.space = task.config_space

        self.fea_type = feature_type
        self.num_threads = num_threads
        self.log_interval = log_interval

        self.fea_type = 'simplified_ast'
        self.num_threads = num_threads
        self.log_interval = log_interval

        self.rnn_params = rnn_params

        if upper_model:
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()
        self.upper_model = upper_model
        self.feature_extra_ct = 0
        self.pool = None
        self.base_model = None

        self._reset_pool(self.space, self.target, self.task)

        self.ctx = mx.gpu() if rnn_params['ctx'] == 'gpu' else mx.cpu()
        self.net = TreeRNNBlock(rnn_params['cell_type'], voc_size=rnn_params['voc_size'],
                                emb_dim=rnn_params['emb_dim'],
                                rnn_hidden_size=rnn_params['rnn_hidden_size'],
                                decoder_hidden_size=rnn_params['decoder_hidden_size'],
                                max_n_children=rnn_params['max_n_children'],
                                num_mem_slots=rnn_params['num_mem_slots'])

        # init parameter and optimizer
        self.net.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.l2loss = gluon.loss.L2Loss()
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam',
                                     {'learning_rate': rnn_params['learning_rate'],
                                      'clip_gradient': rnn_params['clip_gradient'],
                                      'wd': rnn_params['wd']})

        self.learning_rate = rnn_params['learning_rate']
        self.train_batch_size = rnn_params['train_batch_size']
        self.loss_type = rnn_params['loss_type']
        self.name = 'treernn' + "-" + self.loss_type

        self.moving_loss = None

        if upper_model:  # share a same feature cache with upper model
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()

    def save(self, filename):
        self.net.save_params(filename)

    def load(self, filename):
        self.net.load_params(filename, ctx=self.ctx)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""
        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        # use global variable to pass common arguments
        global _extract_space, _extract_target, _extract_task
        _extract_space = space
        _extract_target = target
        _extract_task = task
        self.pool = multiprocessing.Pool(self.num_threads)

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _get_pool(self):
        if self.upper_model:
            return self.upper_model._get_pool()
        return self.pool

    def _base_model_discount(self, sample_size):
        return 1.0 / (2 ** (sample_size / 64.0))

    def _train(self, train_data, plan_size, max_batch):
        tic = time.time()

        # begin training
        net, trainer, l2loss, ctx = self.net, self.trainer, self.l2loss, self.ctx
        trainer.set_learning_rate(self.learning_rate)
        batch_size = self.train_batch_size
        moving_loss = self.moving_loss

        recall = -1e9
        eval_recall = None
        n_batch = 0
        best_metric, best_round = -1e9, 1e9
        while n_batch < max_batch:
            for _, (son, emb_idx, add_fea, label, base) in enumerate(train_data):
                emb_idx, add_fea, label, base = [x.as_in_context(ctx)
                                                 for x in [emb_idx, add_fea, label, base]]
                son = son[0].asnumpy()

                if self.loss_type == 'rank':  # pairwise rank loss
                    size = label.shape[0]
                    with autograd.record():
                        p = net.batch_forward(len(son)-1, son, emb_idx, add_fea).reshape((-1,))
                        z = p + base

                        tmp_l = range(size)
                        tmp_r = [int(x) for x in np.random.randint(size, size=(size,))]

                        index_l, index_r = [], []
                        for l, r in zip(tmp_l, tmp_r):
                            label_l, label_r = label[l], label[r]
                            if label_l == label_r:
                                continue
                            if label_l < label_r:
                                index_l.append(l)
                                index_r.append(r)
                            else:
                                index_l.append(r)
                                index_r.append(l)
                        assert index_l

                        o = -(z[index_l] - z[index_r])
                        loss = -o + mx.nd.log(1 + mx.nd.exp(o))
                elif self.loss_type == 'reg':  # rmse loss
                    with autograd.record():
                        p = net.batch_forward(len(son)-1, son, emb_idx, add_fea).reshape((-1,))
                        z = p + base
                        loss = l2loss(z, label)
                else:
                    raise ValueError("Invalid loss type: " + self.loss_type)

                loss.backward()
                trainer.step(batch_size)

                loss = mx.nd.mean(loss).asscalar()
                moving_loss = loss if moving_loss is None else 0.98 * moving_loss + 0.02 * loss

                n_batch += 1

                # evaluate recall
                if n_batch % self.rnn_params['train_eval_every'] == 0:
                    preds = []
                    labels = []
                    for _, (son_, emb_idx_, add_fea_, label_, base_) in enumerate(train_data):
                        emb_idx_, add_fea_, base_ = [x.as_in_context(ctx)
                                                     for x in [emb_idx_, add_fea_, base_]]
                        son_ = son_[0].asnumpy()

                        z = net.batch_forward(len(son_)-1, son_, emb_idx_, add_fea_)
                        preds.append((mx.nd.squeeze(z) + base_).asnumpy())
                        labels.append(label_.asnumpy())
                    preds, labels = np.concatenate(preds), np.concatenate(labels)
                    recall = average_recall(preds, labels, plan_size)

                if recall > best_metric:
                    best_metric, best_round = recall, n_batch

                if n_batch % self.log_interval == 0:
                    if eval_recall is not None:
                        logger.debug("batch: %d\tmoving loss: %.6f\trecall@%d: %.4f\t"
                                     "eval_recall@%d: %.4f\telapsed: %.2f",
                                     n_batch, np.sqrt(moving_loss), plan_size, recall,
                                     plan_size, eval_recall, time.time() - tic)
                    else:
                        logger.debug("batch: %d\tmoving loss: %.6f\trecall@%d: %.4f\telapsed: %.2f",
                                     n_batch, np.sqrt(moving_loss), plan_size,
                                     recall, time.time() - tic)

            if best_round + self.rnn_params['train_early_stopping'] < n_batch:
                break

        self.moving_loss = moving_loss
        return best_metric, best_round, n_batch

    def fit(self, xs, ys, plan_size):
        tic = time.time()

        self._reset_pool(self.space, self.target, self.task)
        sons, emb_idxs, add_feas = self._get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(ys)
        y_train = y_train / max(y_max, 1e-8)

        # load results from base model
        base_ys = None
        if self.base_model:
            discount = self._base_model_discount(len(xs))
            if discount < 0.05:  # discard base model
                self.base_model.upper_model = None
                self.base_model = None
            else:
                base_ys = self.base_model.predict(xs, output_margin=True)

        if base_ys is None:
            base_ys = np.zeros_like(ys)

        # set data iterators
        emb_idxs_train, add_feas_train, y_train, base_train = \
            [mx.nd.array(x) for x in [emb_idxs, add_feas, y_train, base_ys]]
        train_data = mx.gluon.data.DataLoader(
            mx.gluon.data.ArrayDataset(sons, emb_idxs_train, add_feas_train, y_train, base_train),
            self.train_batch_size, shuffle=True)

        self._train(train_data, plan_size, self.rnn_params['max_batch'])

        logger.debug("TreeGRU train: %.2f\tobs: %d\tn_cache: %d",
                     time.time() - tic, len(xs),
                     self.feature_cache.size(self.fea_type))

    def fit_log(self, records, plan_size):
        raise RuntimeError()

    def predict(self, xs, output_margin=False):
        """predict for a single batch"""
        sons, emb_idxs, add_feas = self._get_feature(xs)

        net, ctx = self.net, self.ctx
        emb_idxs, add_feas = [mx.nd.array(x, ctx=ctx) for x in [emb_idxs, add_feas]]
        sons = sons[0]
        preds = net.batch_forward(len(sons) - 1, sons, emb_idxs, add_feas)
        return preds.asnumpy().flatten()

    def _get_feature(self, indexes):
        """get features for a batch of indexes, run extraction if we do not have cache for them

        Parameters
        ----------
        indexes: Array of int
            indexes of configurations in the space
        """
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 50000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            feas = self._get_pool().map(_extract_ast, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea

        son, emb_idx, add_fea = fea_cache[indexes[0]]

        sons = np.empty((len(indexes),) + son.shape, np.int32)
        emb_idxs = np.empty((len(indexes),) + emb_idx.shape, np.int32)
        add_feas = np.empty((len(indexes),) + add_fea.shape, np.float32)
        for i, ii in enumerate(indexes):
            fea = fea_cache[ii]
            sons[i] = fea[0]
            emb_idxs[i] = fea[1]
            add_feas[i] = fea[2]

        return sons, emb_idxs, add_feas

    def __del__(self):
        self._close_pool()

_extract_space = None
_extract_target = None
_extract_task = None
_extract_add_stats = False

def _extract_ast(index):
    """extract feature for an index in config space"""
    config = _extract_space.get(index)
    with _extract_target:
        sch, arg_bufs = _extract_task.instantiate(config)
    son, emb_idx, add_fea = get_simplified_ast(sch, arg_bufs,
                                               take_log=True,
                                               add_stats=_extract_add_stats)
    # add as additional feature of root
    values = list(config.get_other_option().values())
    if values:
        add_fea[-1][-len(values):] = values

    return son, emb_idx, add_fea

def _extract_ast_log(arg):
    """extract feature from log dataset"""
    inp, res = arg
    config = inp.config
    with inp.target:
        sch, arg_bufs = inp.task.instantiate(config)
    son, emb_idx, add_fea = get_simplified_ast(sch, arg_bufs,
                                               take_log=True,
                                               add_stats=_extract_add_stats)
    # add as additional feature of root
    values = list(config.get_other_option().values())
    if values:
        add_fea[-1][-len(values):] = values

    if res.error_no == 0:
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0

    return son, emb_idx, add_fea, y
