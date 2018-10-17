"""experiment code to study cost models"""

import argparse
import time
import logging
import time
import pickle
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import mxnet as mx
from mxnet import gluon, autograd

import tvm
import topi
from tvm import autotvm
from tvm.autotvm.task.nnvm_integration import TaskExtractEnv
from tvm.autotvm.tuner.metric import recall_curve, cover_curve, max_curve, get_rank, average_recall
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel,\
    _extract_itervar_feature_log, _extract_curve_feature_log, _extract_knob_feature_log, \
    custom_callback, xgb_average_recalln_curve_score
from tvm.autotvm.tuner.treernn_cost_model import _extract_ast_log, TreeRNNBlock


def generate_tune_packs(item_list):
    """generate grid search parameters"""
    ret = []
    now = {}

    def dfs(depth):
        if depth == len(item_list):
            ret.append(now.copy())
            return

        name = item_list[depth][0]
        for value in item_list[depth][1]:
            now[name] = value
            dfs(depth + 1)
    dfs(0)
    return ret

N_TRIAL = 500
TOP_K_PERCENT = 0.001

TEST_SIZE_MUTIPLIER = 5

def plot_curve(preds, labels):
    trials = np.argsort(preds)[::-1]
    labels = labels / np.max(labels)
    scores = labels[trials]
    print("MAX:", np.max(scores[:N_TRIAL]))
    ranks = get_rank(labels[trials])

    top_k = max(int(TOP_K_PERCENT * len(preds)), 1)
    plt.plot((max_curve(scores) / np.max(labels))[:N_TRIAL])
    plt.plot(recall_curve(ranks, top_k)[:N_TRIAL])
    plt.plot(recall_curve(ranks)[:N_TRIAL])
    plt.plot(cover_curve(ranks)[:N_TRIAL])

    plt.legend(["max", "recall-%d" % top_k, "recall-n", "cover"])
    plt.yticks(np.arange(0, 2, 0.1))
    plt.ylim([0, 1])
    plt.xlim([0, N_TRIAL])
    plt.xlabel("number of trials")
    plt.grid()
    plt.show()


def evaluate_rmse(net, data_iter, ctx):
    rmse = mx.metric.RMSE()
    for i, data in enumerate(data_iter):
        data = [x.as_in_context(ctx) for x in data]
        output = net(*data[:-1])
        rmse.update(preds=output, labels=data[-1])
    return rmse.get()[1]

def train_single_mlp(x_train, y_train, x_test, y_test):
    y_max = np.max(y_train)
    y_train /= y_max
    y_test /= y_max

    n_epoch = 5000
    batch_size = 32
    learning_rate = 1e-4
    wd = 5e-4
    print_every = 20
    early_stopping = 100

    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(512, activation='sigmoid'))
        net.add(gluon.nn.Dense(256, activation='relu'))
        net.add(gluon.nn.Dense(256, activation='sigmoid'))
        net.add(gluon.nn.Dense(1))

    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    l2loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'clip_gradient': 5.0,
                             'wd': wd})

    x_train, y_train = [mx.nd.array(x) for x in [x_train, y_train]]
    x_test, y_test = [mx.nd.array(x) for x in [x_test, y_test]]
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(x_train, y_train),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(x_test, y_test),
                                         512, shuffle=False)

    tic = time.time()
    eval_time = 0
    min_loss = 1e9
    min_round = 0
    for e in range(n_epoch):
        rmse = mx.metric.RMSE()
        for i, (data, label) in enumerate(train_data):
            data, label = [x.as_in_context(ctx) for x in [data, label]]
            with autograd.record():
                output = net(data)
                loss = l2loss(output, label)
            loss.backward()
            rmse.update(output, label)
            trainer.step(len(data))
        train_rmse = rmse.get()[1]

        if train_rmse < min_loss:
            min_loss = train_rmse
            min_round = e

        if e % print_every == 0 or (e + 1) == n_epoch:
            eval_tic = time.time()
            tmp_index = np.random.choice(np.arange(len(y_test)), TEST_SIZE_MUTIPLIER * len(y_train), replace=False)
            tmp_test = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(x_test[tmp_index], y_test[tmp_index]),
                                                512, shuffle=False)
            eval_rmse = evaluate_rmse(net, tmp_test, ctx)
            eval_time += time.time() - eval_tic
            print("epoch: %d\ttest rmse: %.6f\ttrain rmse: %.6f\ttime per epoch: %.2f\teval: %.2f" %
                  (e, eval_rmse, train_rmse, (time.time() - tic - eval_time)/(e+1), time.time() - eval_tic))

        if e - min_round >= early_stopping:
            break
    print("train done %.2f" % (time.time() - tic))

    tic = time.time()
    preds, labels = [], []
    for data, label in test_data:
        preds.append(net(data.as_in_context(ctx)).asnumpy())
        labels.append(label.asnumpy())
    preds, labels = [np.concatenate(x, axis=0).squeeze() for x in [preds, labels]]
    print("predict done %.2f" % (time.time() - tic))
    plot_curve(preds, labels)


def train_single_treernn(sons_train, emb_idxs_train, add_feas_train, y_train,
                         sons_test, emb_idxs_test, add_feas_test, y_test,
                         loss_type, plan_size):
    y_max = np.max(y_train)
    y_train /= y_max
    y_test /= y_max

    # hyper parameter
    n_epoch = 1000
    batch_size = 64
    eval_batch_size = 1024
    learning_rate = 8e-4
    wd = 1e-6
    print_every = 50
    num_slots = 0

    print("loss type %s\tnum_slots: %d\t" % (loss_type, num_slots))

    net = TreeRNNBlock('gru', voc_size=128, emb_dim=128, rnn_hidden_size=128,
                       decoder_hidden_size=[128], max_n_children=20, num_mem_slots=num_slots,
                       act='sigmoid')

    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    l2loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'clip_gradient': 5.0,
                             'wd': wd})

    emb_idxs_train, add_feas_train, y_train = [mx.nd.array(x) for x in
                                               [emb_idxs_train, add_feas_train, y_train]]
    emb_idxs_test, add_feas_test, y_test = [mx.nd.array(x) for x in
                                            [emb_idxs_test, add_feas_test, y_test]]

    train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(
        sons_train, emb_idxs_train, add_feas_train, y_train),
        batch_size=batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(
        sons_test, emb_idxs_test, add_feas_test, y_test),
        batch_size=eval_batch_size, shuffle=True)

    tic = time.time()
    moving_loss = None
    for e in range(n_epoch):
        for i, (son, emb_idx, add_fea, label) in enumerate(train_data):
            emb_idx, add_fea, label = [x.as_in_context(ctx)
                                       for x in [emb_idx, add_fea, label]]
            son = son[0].asnumpy()  # the tree structures are the same for a batch

            if loss_type == 'rank':  # pairwise rank loss
                size = label.shape[0]
                with autograd.record():
                    z = net.batch_forward(len(son)-1, son, emb_idx, add_fea)

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

                    o = -(z[index_l] - z[index_r])
                    loss = -o + mx.nd.log(1 + mx.nd.exp(o))
            elif loss_type == 'reg':    # rmse loss
                with autograd.record():
                    z = net.batch_forward(len(son)-1, son, emb_idx, add_fea)
                    loss = l2loss(z, label)
            else:
                raise ValueError("Invalid loss type: " + loss_type)

            loss.backward()
            trainer.step(batch_size)

            loss = mx.nd.mean(loss).asscalar()
            moving_loss = loss if moving_loss is None else 0.98 * moving_loss + 0.02 * loss

        # evaluate recall for several random batches on test data
        if e % print_every == 0:
            preds, labels = [], []
            n_batch = 1e9  # random pick 10 batches
            for i, (son, emb_idx, add_fea, label) in enumerate(test_data):
                emb_idx, add_fea = [x.as_in_context(ctx) for x in [emb_idx, add_fea]]
                son = son[0].asnumpy()

                z = net.batch_forward(len(son) - 1, son, emb_idx, add_fea)
                preds.append(z.asnumpy().squeeze())
                labels.append(label.asnumpy().squeeze())

                if i > n_batch:
                    break
            preds, labels = np.concatenate(preds), np.concatenate(labels)
            recall = average_recall(preds, labels, plan_size)

            logging.info("epoch: %d\tmoving loss: %.6f\ttest-a-recall: %.4f\telapsed: %.2f",
                         e, np.sqrt(moving_loss), recall, time.time() - tic)

    print("train done %.2f" % (time.time() - tic))

    # predict on test data
    tic = time.time()
    preds, labels = [], []
    for i, (son, emb_idx, add_fea, label) in enumerate(test_data):
        emb_idx, add_fea, label = [x.as_in_context(ctx) for x in [emb_idx, add_fea, label]]
        son = son[0].asnumpy()

        z = net.batch_forward(len(son) - 1, son, emb_idx, add_fea)
        preds.append(z.asnumpy().squeeze())
        labels.append(label.asnumpy().squeeze())
    print("predict done %.2f" % (time.time() - tic))

    preds, labels = np.concatenate(preds), np.concatenate(labels)
    plot_curve(preds, labels)


##### XGBoost model (regression, rank, binary classficiation) #####
def train_single_xgb_reg(x_train, y_train, x_test, y_test, plan_size):
    xgb_params = {
        'max_depth': 3,
        'gamma': 0.0001,
        'min_child_weight': 1,

        'subsample': 1.0,
        'silent': 1,

        'eta': 0.3,
        'lambda': 1.00,
        'alpha': 0,

        'objective': 'reg:linear',
    }
    y_max = max(np.max(y_train), 1e-8)
    y_train /= y_max
    y_test /= y_max

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test, y_test)

    tic = time.time()
    bst = xgb.train(xgb_params, dtrain,
                    num_boost_round=8000,
                    callbacks=[custom_callback(
                        stopping_rounds=200,
                        metric='tr-a-recall@%d' % plan_size,
                        evals=[(dtrain, 'tr'), (dtest, 'te')],
                        maximize=True,
                        fevals=[
                            xgb_average_recalln_curve_score(plan_size),
                        ],
                        verbose_eval=5)],
                    )
    print("train done %.2f" % (time.time() - tic))

    # plot curves
    preds = bst.predict(dtest)
    plot_curve(preds, y_test)


def train_single_xgb_rank(x_train, y_train, x_test, y_test, plan_size):
    xgb_params = {
        'max_depth': 3,
        'gamma': 0.0001,
        'min_child_weight': 1,

        'subsample': 1.0,
        'silent': 1,

        'eta': 0.3,
        'lambda': 1.00,
        'alpha': 0,

        'objective': 'rank:pairwise',
    }
    y_train /= np.max(y_train)
    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test, y_test)

    tic = time.time()
    bst = xgb.train(xgb_params, dtrain,
                    num_boost_round=8000,
                    callbacks=[custom_callback(
                        stopping_rounds=40,
                        metric='tr-a-recall@%d' % plan_size,
                        evals=[(dtrain, 'tr'), (dtest, 'te')],
                        maximize=True,
                        fevals=[
                            xgb_average_recalln_curve_score(plan_size),
                        ],
                        verbose_eval=5)],
                    )
    print("train done %.2f" % (time.time() - tic))

    # plot curves
    preds = bst.predict(dtest)
    plot_curve(preds, y_test)


##### Extract Feature (flatten, two-level, ast) #####
flatten_feature_extract_func = None
def row2flatten(row):
    return flatten_feature_extract_func(autotvm.record.decode(row))

def extract_flatten(filename, feature_func):
    global flatten_feature_extract_func
    flatten_feature_extract_func = feature_func
    cache_file = filename + ".flatten.cache"
    if os.path.exists(cache_file):
        print("load cached data...")
        xs, ys = pickle.load(open(cache_file, "rb"))
    else:
        print("extract feature...")
        rows = list(open(args.i).readlines())
        xys = autotvm.util.pool_map(row2flatten, rows, batch_size=10000, verbose=True)
        xs, ys = zip(*xys)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        pickle.dump((xs, ys), open(cache_file, "wb"))

    return xs, ys

def row2ast(row):
    return _extract_ast_log(autotvm.record.decode(row))

def extract_ast(filename):
    cache_file = filename + ".ast.cache"

    if not os.path.exists(cache_file):
        print("extract feature...")
        rows = list(open(args.i).readlines())
        a = autotvm.util.pool_map(row2ast, rows, batch_size=5000, verbose=True)

        n_rows = len(a)

        sons = np.array([a[i, 0] for i in range(n_rows)], dtype=np.int16)
        emb_idxs = np.array([a[i, 1] for i in range(n_rows)], dtype=np.int16)
        add_feas = np.array([a[i, 2] for i in range(n_rows)], dtype=np.float32)
        ys = np.array([a[i, 3] for i in range(n_rows)], dtype=np.float32)
        a = None
        del a

        pickle.dump((sons, emb_idxs, add_feas, ys), open(cache_file, "wb"))
    else:
        print("load cached feature")
        sons, emb_idxs, add_feas, ys = pickle.load(open(cache_file, 'rb'))

    sons = np.array(sons, dtype=np.int32)
    emb_idxs = np.array(emb_idxs, dtype=np.int32)
    return sons, emb_idxs, add_feas, ys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, required=True)
    parser.add_argument("--pick", type=int, default=100000000)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0x5D)
    parser.add_argument("--model",
                        choices=["mlp", "treernn-reg", "treernn-rank", "xgb-reg", "xgb-rank", 'xgb-bin'],
                        default='mlp')
    parser.add_argument("--mode", choices=["single"], default='single')
    parser.add_argument("--fea-type", choices=["itervar", "knob", "curve"], default='itervar')
    parser.add_argument("--non-zero", action='store_true', help="discard zero data")
    parser.add_argument("--filter-out", action='store_true', help="filter out some feature")
    args = parser.parse_args()

    ctx = mx.gpu(0)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("autotvm")
    logger.setLevel(logging.DEBUG)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(args)

    # init TOPI tuning tasks
    TaskExtractEnv.get()

    plan_size = 64

    if args.mode == "single":
        if args.model in ['treernn-reg', 'treernn-rank']:
            sons, emb_idxs, add_feas, ys = extract_ast(args.i)

            # remove invalid samples
            non_zero = ys > 1e-8
            print("# zero", len(ys) - np.sum(non_zero))
            if args.non_zero:
                sons, emb_idxs, add_feas, ys = [x[non_zero] for x in [sons, emb_idxs, add_feas, ys]]

            # shuffle and split
            np.random.seed(args.seed)
            shuffle_index = np.random.permutation(len(sons))
            roots, emb_idxs, add_feas, ys = [x[shuffle_index] for x in [sons, emb_idxs, add_feas, ys]]
            sons_train, emb_idxs_train, add_feas_train, y_train = [x[:args.train_size] for
                                                                   x in [roots, emb_idxs, add_feas, ys]]
            sons_test, emb_idxs_test, add_feas_test, y_test = [x[args.train_size:] for
                                                               x in [roots, emb_idxs, add_feas, ys]]

            print("===== Dataset Info =====")
            print("roots_train   ", sons_train.shape)
            print("emb_idxs_train", emb_idxs_train.shape)
            print("add_feas_train", add_feas_train.shape)
            print("y_test        ", y_test.shape)

            loss_type = args.model.split('-')[1]
            train_single_treernn(sons_train, emb_idxs_train, add_feas_train, y_train,
                                 sons_test, emb_idxs_test, add_feas_test, y_test, loss_type,
                                 plan_size)
        else:
            '''
                feature_name = {
                    "_attr_": ["length", "nest_level", "topdown", "bottomup"] +
                              ["ann_%d" % i for i in range(20)],
                    "_arith_": ["add", "mul", "div"],
                    "buf_touch": ["stride", "mod", "count", "reuse", "T_count", "T_reuse"],
                }
            '''
            if args.fea_type == 'itervar':
                feature_func = _extract_itervar_feature_log
            elif args.fea_type == 'knob':
                feature_func = _extract_knob_feature_log
            elif args.fea_type == 'curve':
                feature_func = _extract_curve_feature_log
            else:
                raise ValueError("Invalid feature type: " + args.fea_type)

            xs, ys = extract_flatten(args.i, feature_func)

            # filter out some feature
            if args.filter_out:
                assert args.fea_type == 'itervar'
                names = autotvm.feature.get_flatten_name(open(args.i).readline())
                for i in range(len(names)):
                    # "length", "nest_level", "topdown", "bottomup",
                    # "add", "mul", "div",
                    # "stride", "mod", "count", "reuse", "T_count", "T_reuse"
                    filter_out = True
                    for x in ['length', 'bottomup', 'count', 'stride']:
                        if x in names[i]:
                            filter_out = False
                    if filter_out:
                        xs[:, i] = 0

            # remove invalid samples
            non_zero = ys > 1e-6
            print("# zero", len(ys) - np.sum(non_zero))

            if args.non_zero:
                xs, ys = xs[non_zero], ys[non_zero]

            # shuffle and split
            np.random.seed(args.seed)
            shuffle_index = np.random.permutation(len(xs))
            xs, ys = xs[shuffle_index][:args.pick], ys[shuffle_index][:args.pick]
            x_train, y_train = xs[:args.train_size], ys[:args.train_size]
            x_test, y_test = xs[args.train_size:], ys[args.train_size:]

            print("===== Dataset Info =====")
            print("x_train", x_train.shape)
            print("y_test", y_test.shape)

            if args.model == "xgb-reg":
                train_single_xgb_reg(x_train, y_train, x_test, y_test, plan_size)
            elif args.model == 'xgb-rank':
                train_single_xgb_rank(x_train, y_train, x_test, y_test, plan_size)
            elif args.model == 'mlp':
                train_single_mlp(x_train, y_train, x_test, y_test)
            else:
                raise ValueError("invalid model " + args.model)
