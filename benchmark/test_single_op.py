"""Operators that we want to solve"""

import argparse
import tvm
from tvm import autotvm


def matmul(N, L, M, dtype='float32'):
    A = tvm.placeholder((N, L), dtype=dtype, name='A')
    B = tvm.placeholder((L, M), dtype=dtype, name='B')

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j:
                    tvm.sum(A[i, k] * B[k, j], axis=k), name='C')

    return C

def batch_matmul(B, N, L, M, dtype='float32'):
    A = tvm.placeholder((B, N, L), dtype=dtype, name='A')
    B = tvm.placeholder((B, L, M), dtype=dtype, name='B')

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((B, N, M), lambda b, i, j:
                    tvm.sum(A[b, i, k] * B[b, k, j], axis=k), name='C')

    return C


def conv3d(N, CI, D, H, W, CO, kernel_size, stride, padding, dtype='float32'):
    KD = KH = KW = kernel_size
    SD = SH = SW = stride
    PD = PH = PW = padding

    A = tvm.placeholder((N, CI, D, H, W), dtype=dtype, name='A')
    B = tvm.placeholder((CO, CI, KD, KH, KW), dtype=dtype, name='B')

    ci = tvm.reduce_axis((0, CI), 'rc')
    kd = tvm.reduce_axis((0, KD), 'kd')
    kh = tvm.reduce_axis((0, KH), 'kh')
    kw = tvm.reduce_axis((0, KW), 'kw')

    OD = (D + 2 * PD - KD) // SD + 1
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    C = tvm.compute((N, CO, OD, OH, OW), lambda n, co, od, oh, ow:
                    tvm.sum(A[n, ci, od * SD + kd, oh * SH + kh, ow * SW + kw] *
                            B[co, ci, kd, kh, kw], axis=[ci, kd, kh, kw]), name='C')

    return C

# def maxpool_3d(N, CI, H, W, ):


def dilated_conv2d(N, CI, H, W, CO, kernel_size, stride, padding, dilation, dtype='float32'):
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    DH = DW = dilation

    A = tvm.placeholder((N, CI, H, W), dtype=dtype, name='A')
    B = tvm.placeholder((CO, CI, KH, KW), dtype=dtype, name='B')

    ci = tvm.reduce_axis((0, CI), 'rc')
    kh = tvm.reduce_axis((0, KH), 'kh')
    kw = tvm.reduce_axis((0, KW), 'kw')

    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    C = tvm.compute((N, CO, OH, OW), lambda n, co, oh, ow:
                    tvm.sum(A[n, ci, oh * SH + kh, ow * SW + kw] *
                            B[co, ci, kh, kw], axis=[ci, kh, kw]), name='C')

    return C

def norm(D1, D2, D3, dtype='float32'):
    A = tvm.placeholder((D1, D2, D3), dtype=dtype, name='A')

    r1 = tvm.reduce_axis((0, D1), name='r1')
    r2 = tvm.reduce_axis((0, D2), name='r2')
    r3 = tvm.reduce_axis((0, D3), name='r3')

    C = tvm.compute((1,), lambda i: tvm.sum(A[r1, r2, r3] * A[r1, r2, r3], axis=[r1, r2, r3]), name='C')

    return C

