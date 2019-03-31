import argparse
import torch
import timeit

def benchmark(setup, stmt, min_repeat_second=0.5):
    total_time = 0
    n_repeat = 10

    while total_time < min_repeat_second:
        total_time = timeit.timeit(setup=setup, stmt=stmt, number=n_repeat)
        n_repeat = int((min_repeat_second / total_time) * n_repeat) + 1

    return total_time / n_repeat

dtype = 'float32'

def np_norm(N):
    t = benchmark(setup='import numpy as np\n' + 
                        'A = np.random.randn(%d,%d,%d).astype("%s")\n' % (N, N, N, dtype),
                  stmt= 'B = np.linalg.norm(A)')

    return t, 2 * N * N * N


def np_matmul(N, M, L):
    t = benchmark(setup='import numpy as np\n' + 
                        'A = np.random.randn(%d,%d).astype("%s")\n' % (N, L, dtype) +
                        'B = np.random.randn(%d,%d).astype("%s")\n' % (L, M, dtype),
                  stmt= 'C = np.matmul(A, B)')

    return t, 2 * N * M * L


def torch_norm(N):
    t = benchmark(setup='import torch\n' + 
                        'A = torch.rand(%d,%d,%d)\n' % (N, N, N),
                  stmt= 'C = torch.norm(A)')

    return t, 2 * N * N * N


def torch_matmul(N, M, L):
    t = benchmark(setup='import torch\n' + 
                        'A = torch.randn(%d,%d)\n' % (N, L) +
                        'B = torch.randn(%d,%d)\n' % (L, M),
                  stmt= 'C = torch.matmul(A, B)')

    return t, 2 * N * L * M


def torch_conv3d(N, CI, D, H, W, CO, kernel_size, stride, padding, dilation):
    KD = KH = KW = kernel_size
    SD = SH = SW = stride
    PD = PH = PW = padding
    OD = (D + 2 * PD - KD) // SD + 1
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    t = benchmark(setup='import torch\n' + 
                        'A = torch.rand(%d,%d,%d,%d,%d)\n' % (N, CI, D, H, W) +
                        'B = torch.rand(%d,%d,%d,%d,%d)\n' % (CO, CI, KD, KH, KW),
                  stmt= 'C = torch.nn.functional.conv3d(A, B, stride=%d, padding=%d, dilation=%d)'
                         % (stride, padding, dilation))

    return t, 2 * N * CO * OD * OH * OW * CI * KD * KH * KW


def torch_avg_pool3d(N, CI, D, H, W, kernel_size, stride, padding):
    KD = KH = KW = kernel_size
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (D + 2 * PD - KD) // SD + 1
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    t = benchmark(setup='import torch\n' + 
                        'A = torch.rand(%d,%d,%d,%d,%d)\n' % (N, CI, D, H, W),
                  stmt= 'C = torch.nn.functional.avg_pool3d(A, kernel_size=%d, stride=%d, padding=%d)'
                         % (kernel_size, stride, padding))

    A = torch.rand(N, CI, D, H, W)
    C = torch.nn.functional.avg_pool3d(A, kernel_size=kernel_size, stride=stride, padding=padding)

    return t, 2 * N * CI * OD * OH * OW * KD * KH * KW


def torch_conv2d(N, CI, H, W, CO, kernel_size, stride, padding, dilation):
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    DH = DW = dilation

    dilated_KH = (KH - 1) * DH + 1
    dilated_KW = (KW - 1) * DW + 1
    OH = (H + 2 * PH - dilated_KH) // SH + 1
    OW = (W + 2 * PW - dilated_KW) // SW + 1


    t = benchmark(setup='import torch\n' + 
                        'A = torch.rand(%d,%d,%d,%d)\n' % (N, CI, H, W) +
                        'B = torch.rand(%d,%d,%d,%d)\n' % (CO, CI, KH, KW),
                  stmt= 'C = torch.nn.functional.conv2d(A, B, stride=%d, padding=%d, dilation=%d)'
                         % (stride, padding, dilation))

    return t, 2 * N * CO * CI * OH * OW * KH * KW


test_cases = [
    ("numpy", "norm", np_norm, (256,)),
    ("numpy", "matmul", np_matmul, (1024, 1024, 1024)),

    ("torch", "norm",   torch_norm, (256,)),
    ("torch", "matmul", torch_matmul, (1024, 1024, 1024)),
    ("torch", "conv3d", torch_conv3d, (1, 64, 16, 56, 56, 64, 3, 1, 0, 1)),
    ("torch", "avg_pool3d", torch_avg_pool3d, (1, 2048, 7, 7, 7, 7, 1, 0)),
    ("torch", "dilated_conv2d", torch_conv2d, (1, 64, 56, 56, 64, 3, 1, 0, 16)),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["numpy", "torch", "all"], default="torch")
    parser.add_argument("--op", type=str)
    args = parser.parse_args()

    if args.backend == "all":
        backends = ["numpy", "torch"]
    else:
        backends = [args.backend]

    for case in test_cases:
        backend, name, func, func_args = case

        if backend not in backends:
            continue

        if args.op is not None and name != args.op:
            continue

        cost, flop = func(*func_args)

        record = ("%s\t%s\t%.3f\t%.2f" % (backend, name, cost * 1e3, (flop / 1e9) / cost))

        print(record)

