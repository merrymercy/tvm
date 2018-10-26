# Experiments in the autotvm paper

## Install
1. Install tvm [help](https://docs.tvm.ai/install/from_source.html) with CUDA and LLVM enabled.
2. Install other missing dependencies (redis, mxnet) by `pip install xxx`

## Setup RPC Tracker and Devices
1. Open one terminal to run the RPC tracker

```bash
python3 -m tvm.exec.rpc_tracker
```

2. Open another terminal to run the RPC Server

```bash
python3 -m tvm.exec.rpc_server --tracker localhost:9190 --key titanx
```

3. Confirm the connection

```bash
python3 -m tvm.exec.query_rpc_tracker
```

You are supposed to see a free "titanx" in the queue status.

## Test tuners

### Evaluate the Cost Models
```bash
python3 test_tuners.py --tuner xgb-rank --n-trial 800
python3 test_tuners.py --tuner treegru-rank --n-trail 800
python3 test_tuners.py --tuner ga --n-trial 2000
python3 test_tuners.py --tuner random --n-trial 2000

python3 plot_cost_models.py
```

### Evaluate the Rank and Regression Loss Function
```bash
python3 test_tuners.py --tuner xgb-rank --n-trial 800
python3 test_tuners.py --tuner treegru-rank --n-trail 800
python3 test_tuners.py --tuner xgb-reg --n-trial 800
python3 test_tuners.py --tuner treegru-reg --n-trial 800

python3 plot_rank_reg.py
```

### Evaluate the Diversity-aware Exploration
```bash
python3 test_tuners.py --tuner xgb-rank --n-trial 800
python3 test_tuners.py --tuner xgb-rank-d2 --n-trail 800
python3 test_tuners.py --tuner xgb-rank-d4 --n-trial 800

python3 plot_diversity.py
```


### Evaluate the Uncertainty-aware Exploration
```bash
python3 test_tuners.py --tuner xgb-reg-mean --n-trial 800
python3 test_tuners.py --tuner xgb-reg-ei --n-trail 800
python3 test_tuners.py --tuner xgb-reg-ucb --n-trial 800

python3 plot_uncertainty.py
```

### Evaluate the Eps-greedy Policy
```bash
python3 test_tuners.py --tuner xgb-rank--n-trial 800
python3 test_tuners.py --tuner xgb-rank-no-eps --n-trail 800

python3 plot_eps_greedy.py
```

