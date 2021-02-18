import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload
def dense_layer(N):
    data = te.placeholder((N, N), name='A')
    kernel = te.placeholder((N, N), name="kernel")
    out = topi.nn.dense(data, kernel)
    return [data, kernel, out]

target = tvm.target.Target("cuda")
N = 2048
task = auto_scheduler.SearchTask(
    func=dense_layer, args=(N,), target=target
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "matmul.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

#task.tune(tune_option)
sch, args = task.apply_best(log_file)
del measure_ctx
func = tvm.build(sch, args, target)


# Check correctness
data_np = np.random.uniform(size=(N, N)).astype(np.float32)
weight_np = np.random.uniform(size=(N, N)).astype(np.float32)
out_np = np.matmul(data_np, weight_np.T)
ctx = tvm.gpu()
data_tvm = tvm.nd.array(data_np, ctx=ctx)
weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
out_tvm = tvm.nd.array(np.ones_like(data_np), ctx=ctx)
func(data_tvm, weight_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, ctx, number=100, repeat=10)
time = np.median(evaluator(data_tvm, weight_tvm, out_tvm).results)
print("shape", data_np.shape, weight_np.shape)
print("Execution time of this operator: %.3f ms" % (time * 1000))
print("Speed: %.3f TFLOPS" % (2 * (N**3) / time / 1e12))

