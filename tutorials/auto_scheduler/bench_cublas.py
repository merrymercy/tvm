import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from tvm.contrib import cublas
target = tvm.target.Target('cuda')

N = 2048
A = te.placeholder((N, N), name='data', dtype='float32')
B = te.placeholder((N, N), name='kernel', dtype='float32')
C = cublas.matmul(A, B, False, True, dtype='float32')

sch = te.create_schedule(C.op)
args = [A, B, C]
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

