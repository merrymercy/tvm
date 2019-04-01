"""
How to use auto scheduler
=========================================================================
**Author**: `Lianmin Zheng <https://https://github.com/merrymercy>`_

In TVM, we decouple the kernel implementation into compute and schedule.
The schedule part requires expert knowledge to provide the best performance.
Optimizing the schedule part is partly automated by the autotvm package, which
takes in a tunable template and try to find good values for the knobs in the template.
However, it still requires a human designed template.

In this tutorial, we present a fully automated auto scheduler. It generates
the tunable template directly from compute declaration. Then this template can either
be

1. statically filled with heuristic rules to get reasonable performance or
2. tuned with some time to get better performance.

We will show two static scheduling examples for cpu and gpu, and one tuning example.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make tvm run faster in tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import tvm
from tvm import autotvm
import topi

######################################################################
# Static scheduling
# -----------------
# In this section, we use the auto-scheduler statically without tuning.
# The auto-scheduler can fill the tunable template according to heuristic rules
# and hardware configurations (set in :code:`autotvm.AutoSchedulerOptions`).
# Take a convolution op for example.

N, C, H, W, CI, CO, KH, KW = 8, 128, 58, 58, 128, 128, 3, 3

A = tvm.placeholder((N, CI, H, W), name='A')
B = tvm.placeholder((CO, CI, KH, KW), name='B')
bias = tvm.placeholder((CO,), name='bias')

ci = tvm.reduce_axis((0, CI), 'rc')
kh = tvm.reduce_axis((0, KH), 'kh')
kw = tvm.reduce_axis((0, KW), 'kw')

OH = H - 2
OW = W - 2

C = tvm.compute((N, CO, OH, OW), lambda n, co, oh, ow:
                tvm.sum(A[n, ci, oh + kh, ow + kw] * B[co, ci, kh, kw],
                        axis=[ci, kh, kw]), name='C')
O = tvm.compute((N, CO, OH, OW), lambda n, co, oh, ow:
                 C[n, co, oh, ow] + bias[co], name='output')

######################################################################
# The schedule API is very simple. We only need to call
# :code:`autotvm.create_schedule`. The arguments are input/output buffers.
# The return values are the created schedule and new buffers.
# The returned buffers may be different from the input buffers because some optimizations
# in the auto scheduler may rewrite the compute declaration.
# But the returned buffers and argument buffers match exactly in shapes and orders.

with tvm.target.create('llvm'):
    s, bufs = autotvm.create_schedule([A, B, bias, O])

# We can inspect the lower ir to see what the auto scheduler does:
# The code is parallelized, vectorize and tiled for CPU.
print(tvm.lower(s, bufs, simple_mode=True))

######################################################################
# Currently, the auto-scheduler supports CPU and GPU backend.
# We can change the target context to GPU.
with tvm.target.create('cuda'):
    s, bufs = autotvm.create_schedule([A, B, bias, O])

print(tvm.lower(s, bufs, simple_mode=True))
# The two compute declarations are fused and dispatched to block/thread correctly.

######################################################################
# Dynamic auto-tuning
# -------------------
# If you have some time budget and want to get better performance. You can autotune
# the generated template as autotune the existing templates in TOPI.
#
# Here we use an image blur function as example.
# You can control the granularity of tuning by setting :code:`tuning_level` in
# :code:`autotvm.AutoSchedulerOptions`. Higher level will enable
# more tunable knobs. For this example, we set :code:`tuning_level` to highest level 3
# to tune the compute locations.

@autotvm.template
def blur():
    A = tvm.placeholder((18, 18), name='A')
    B = tvm.compute((18, 16), lambda i, j: (A[i][j] + A[i][j+1] + A[i][j+2])/3, name='B')
    C = tvm.compute((16, 16), lambda i, j: (B[i][j] + B[i+1][j] + B[i+2][j])/3, name='C')

    with autotvm.AutoScheduleOptions(tuning_level=3): # tuning level 3, the highest level
        s, bufs = autotvm.create_schedule([A, C])

    return s, bufs

measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=1))

tsk = autotvm.task.create(blur, args=(), target='llvm')

# Inspect the configuration space.
# This blur function is relatively simple, so the space only contains
# vectorization knobs and compute location knobs.
print(tsk.config_space)

######################################
# Make 8 random trials and apply the best configuration
tuner = autotvm.tuner.RandomTuner(tsk)
tuner.tune(n_trial=8, measure_option=measure_option)

with tvm.target.create('llvm'):
    s, bufs = tsk.instantiate(tuner.best_config)
    print(tvm.lower(s, bufs, simple_mode=True))
