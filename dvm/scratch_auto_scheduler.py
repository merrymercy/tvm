import tvm
import argparse
import topi
from tvm.contrib.util import get_lower_ir
from tvm import autotvm



#@autotvm.template
def conv2d(N, CI, H, W, CO, kernel_size, stride, padding, dilation, dtype='float32'):
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    DH = DW = dilation

    A = tvm.placeholder((N, CI, H, W), dtype=dtype, name='A')
    B = tvm.placeholder((CO, CI, KH, KW), dtype=dtype, name='B')
    bias = tvm.placeholder((CO,), dtype=dtype, name='bias')

    ci = tvm.reduce_axis((0, CI), 'rc')
    kh = tvm.reduce_axis((0, KH), 'kh')
    kw = tvm.reduce_axis((0, KW), 'kw')

    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    pad_a = tvm.compute((N, CI, H + PH * 2, W + PW * 2), lambda n, ci, h, w:
                        tvm.select(tvm.all(h >= PH, h < H + PH, w >= PW, w < W + PW), A[n, ci, h - PH, w - PW], 0.0),
                        name="pad")
    C = tvm.compute((N, CO, OH, OW), lambda n, co, oh, ow:
                    tvm.sum(pad_a[n, ci, oh * SH + kh * DH, ow * SW + kw * DW] *
                            B[co, ci, kh, kw], axis=[ci, kh, kw]), name='C')
    C = tvm.compute((N, CO, OH, OW), lambda n, co, oh, ow: tvm.select(C[n, co, oh, ow] > 0, C[n, co, oh, ow], 0.0),
                    name="relu")
    C = tvm.compute((N, CO, OH, OW), lambda n, co, oh, ow: C[n, co, oh, ow] + bias[co],
                    name="bias_add")

    s = autotvm.create_schedule([A, B, bias, C])

    print(get_lower_ir(s))

    return s, [A, B, bias, C]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with tvm.target.arm_cpu():
        # A = tvm.placeholder((10, 2000,), name='A')
        # D = topi.argmin(A, axis=1)

        A = tvm.placeholder((2, 56, 64, 64), name='A')
        #D = topi.nn.global_pool(A, 'max')

        D = topi.nn.pool(A, (2, 2), (1, 1), (0, 0, 0, 0), 'max')

        print(tvm.lower(tvm.create_schedule([D.op]), [A, D], simple_mode=True))

        s = autotvm.create_schedule([A, D])
        print(tvm.lower(s, [A, D], simple_mode=True))
        exit()

    N, CI, H, W, CO, kernel_size, stride, padding, dilation = 1, 64, 56, 56, 64, 3, 1, 1, 1
    with tvm.target.cuda():
        conv2d(N, CI, H, W, CO, kernel_size, stride, padding, dilation)

