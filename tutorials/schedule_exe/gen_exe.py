"""
Usage:
# Generate exercises
python3 gen_exe.py > exercise.txt

# Generate exercises with reference solution
python3 gen_exe.py --solution > exercise_with_solution.txt
"""

import argparse
import inspect
import re

import tvm
from tvm import te

EXE_REGISTRY = []


def exe(fun):
    EXE_REGISTRY.append(fun)


def pretty_print_func(func):
    src = '\n'.join(inspect.getsource(func).split('\n')[:])
    src = re.sub('^    ', '', src, flags=re.MULTILINE)
    return src


def pretty_print_ir(ir):
    ir = str(ir)
    ir = "\n".join(str(ir).split('\n')[:-1])
    ir = ir.replace('(float32*)', '')
    return ir


@exe
def single_split():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.compute((128, 128), lambda i, j: A[j][i], name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        i, j = B.op.axis
        io, ii = s[B].split(i, 8)
        return s


    return compute, schedule


@exe
def multi_split():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.compute((128, 128), lambda i, j: A[j][i], name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        i, j = B.op.axis
        io, ii = s[B].split(i, 64)
        io, ii = s[B].split(ii, 32)
        io, ii = s[B].split(ii, 4)
        io, ii = s[B].split(ii, 2)

        return s

    return compute, schedule


@exe
def fuse_reorder():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.compute((128, 128), lambda i, j: A[j][i], name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        i, j = B.op.axis
        s[B].reorder(j, i)
        s[B].fuse(j, i)

        return s

    return compute, schedule


@exe
def parallel_vecotrize():
    def compute():
        A = te.placeholder((128, 16), name='A')
        B = te.compute((128, 16), lambda i, j: A[i][j] + 1, name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        i, j = B.op.axis
        s[B].parallel(i)
        s[B].vectorize(j)

        return s

    return compute, schedule


@exe
def multi_annotations():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.compute((128, 128), lambda i, j: A[i][j] + 1, name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        i, j = B.op.axis
        s[B].parallel(i)
        jo, ji = s[B].split(j, 16)
        s[B].unroll(jo)
        s[B].vectorize(ji)

        return s

    return compute, schedule


@exe
def compute_at():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.compute((128, 126), lambda i, j: (A[i][j] + A[i][j+1] + A[i][j+2]), name='B')
        C = te.compute((126, 126), lambda i, j: (B[i][j] + B[i+1][j] + B[i+2][j]), name='C')

        return [A, B, C]

    def schedule(bufs):
        A, B, C = bufs
        s = te.create_schedule([C.op])

        s[B].compute_at(s[C], s[C].op.axis[1])

        return s

    return compute, schedule


@exe
def nested_compute_at():
    def compute():
        A = te.placeholder((128, 128, 128), name='A')
        B = te.compute((128, 128, 128), lambda i, j, k: A[i][j][k] + 1, name='B')
        C = te.compute((128, 128, 128), lambda i, j, k: B[i][j][k] * 2, name='C')
        D = te.compute((128, 128, 128), lambda i, j, k: C[i][j][k] - 3, name='D')

        return [A, B, C, D]

    def schedule(bufs):
        A, B, C, D = bufs
        s = te.create_schedule([D.op])

        s[C].compute_at(s[D], s[D].op.axis[1])
        s[B].compute_at(s[D], s[D].op.axis[0])

        return s

    return compute, schedule


@exe
def tiled_matmul():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

        return [A, B, C]

    def schedule(bufs):
        A, B, C = bufs
        s = te.create_schedule([C.op])

        i, j = C.op.axis
        k, = C.op.reduce_axis
        io, ii = s[C].split(i, 8)
        jo, ji = s[C].split(j, 8)
        s[C].reorder(io, jo, k, ii, ji)
        return s


    return compute, schedule


@exe
def tiled_matmul_optimized():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

        return [A, B, C]

    def schedule(bufs):
        A, B, C = bufs
        s = te.create_schedule([C.op])

        i, j = C.op.axis
        k, = C.op.reduce_axis
        io, ii = s[C].split(i, 8)
        jo, ji = s[C].split(j, 8)
        s[C].reorder(io, jo, k, ii, ji)

        s[C].unroll(ii)
        s[C].vectorize(ji)
        s[C].parallel(io)

        return s


    return compute, schedule


@exe
def tiled_matmul_fusion():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        bias = te.placeholder((128,), name='bias')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
        D = te.compute((128, 128), lambda i, j: C[i][j] + bias[j], name='D')

        return [A, B, bias, C, D]

    def schedule(bufs):
        A, B, bias, C, D = bufs
        s = te.create_schedule([D.op])

        i, j = C.op.axis
        k, = C.op.reduce_axis
        io, ii = s[C].split(i, 16)
        jo, ji = s[C].split(j, 16)
        s[C].reorder(io, jo, k, ii, ji)

        i, j = D.op.axis
        io, ii = s[D].split(i, 16)
        jo, ji = s[D].split(j, 16)
        s[D].reorder(io, jo, ii, ji)
        s[C].compute_at(s[D], jo)

        return s

    return compute, schedule


@exe
def tiled_matmul_fusion_multi_level():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        bias = te.placeholder((128,), name='bias')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
        D = te.compute((128, 128), lambda i, j: C[i][j] + bias[j], name='D')

        return [A, B, bias, C, D]

    def schedule(bufs):
        A, B, bias, C, D = bufs
        s = te.create_schedule([D.op])

        i, j = C.op.axis
        k, = C.op.reduce_axis
        i0, i1 = s[C].split(i, 16)
        i1, i2 = s[C].split(i1, 4)
        j0, j1 = s[C].split(j, 16)
        j1, j2 = s[C].split(j1, 4)
        k0, k1 = s[C].split(k, 8)
        s[C].reorder(i0, j0, k0, i1, j1, k1, i2, j2)

        i, j = D.op.axis
        i0, i1 = s[D].split(i, 16)
        j0, j1 = s[D].split(j, 16)
        s[D].reorder(i0, j0, i1, j1)
        s[C].compute_at(s[D], j0)

        return s

    return compute, schedule


@exe
def tiled_matmul_cache_write():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

        return [A, B, C]

    def schedule(bufs):
        A, B, C = bufs
        s = te.create_schedule([C.op])

        C_local = s.cache_write(C, 'local')

        i, j = C_local.op.axis
        k, = C_local.op.reduce_axis
        i0, i1 = s[C_local].split(i, 16)
        j0, j1 = s[C_local].split(j, 16)
        s[C_local].reorder(i0, j0, k, i1, j1)

        i, j = C.op.axis
        i0, i1 = s[C].split(i, 16)
        j0, j1 = s[C].split(j, 16)
        s[C].reorder(i0, j0, i1, j1)
        s[C_local].compute_at(s[C], j0)

        return s

    return compute, schedule


@exe
def tiled_matmul_cache_read():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

        return [A, B, C]

    def schedule(bufs):
        A, B, C = bufs
        s = te.create_schedule([C.op])

        A_local = s.cache_read(A, 'local', [C])
        B_local = s.cache_read(B, 'local', [C])
        C_local = s.cache_write(C, 'local')

        i, j = C_local.op.axis
        k, = C_local.op.reduce_axis
        i0, i1 = s[C_local].split(i, 16)
        j0, j1 = s[C_local].split(j, 16)
        s[C_local].reorder(i0, j0, k, i1, j1)

        s[A_local].compute_at(s[C_local], k)
        s[B_local].compute_at(s[C_local], k)

        i, j = C.op.axis
        i0, i1 = s[C].split(i, 16)
        j0, j1 = s[C].split(j, 16)
        s[C].reorder(i0, j0, i1, j1)

        s[C_local].compute_at(s[C], j0)

        return s

    return compute, schedule

@exe
def tiled_matmul_gpu():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

        return [A, B, C]

    def schedule(bufs):
        A, B, C = bufs
        s = te.create_schedule([C.op])

        i, j = C.op.axis
        k, = C.op.reduce_axis
        i0, i1 = s[C].split(i, 16)
        i1, i2 = s[C].split(i1, 4)
        j0, j1 = s[C].split(j, 16)
        j1, j2 = s[C].split(j1, 4)
        s[C].reorder(i0, j0, i1, j1, k, i2, j2)
        s[C].bind(i0, te.thread_axis("blockIdx.x"))
        s[C].bind(j0, te.thread_axis("blockIdx.y"))
        s[C].bind(i1, te.thread_axis("threadIdx.x"))
        s[C].bind(j1, te.thread_axis("threadIdx.y"))
        return s

    return compute, schedule


@exe
def inline():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.compute((128, 128), lambda i, j: A[i][j] + 1, name='B')
        C = te.compute((128, 128), lambda i, j: B[i][j] * 2, name='C')
        D = te.compute((128, 128), lambda i, j: te.exp(C[i][j]), name='D')

        return [A, B, C, D]

    def schedule(bufs):
        A, B, C, D = bufs
        s = te.create_schedule([D.op])

        s[B].compute_inline()
        s[C].compute_inline()

        return s

    return compute, schedule


@exe
def tiled_matmul_fusion_inline():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.placeholder((128, 128), name='B')
        bias = te.placeholder((128,), name='bias')
        k = te.reduce_axis((0, 128), name='k')
        C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
        D = te.compute((128, 128), lambda i, j: C[i][j] + bias[j], name='D')
        E = te.compute((128, 128), lambda i, j: D[i][j] * 2, name='E')

        return [A, B, bias, C, D, E]

    def schedule(bufs):
        A, B, bias, C, D, E = bufs
        s = te.create_schedule([E.op])

        i, j = C.op.axis
        k, = C.op.reduce_axis
        io, ii = s[C].split(i, 16)
        jo, ji = s[C].split(j, 16)
        s[C].reorder(io, jo, k, ii, ji)

        i, j = E.op.axis
        io, ii = s[E].split(i, 16)
        jo, ji = s[E].split(j, 16)
        s[E].reorder(io, jo, ii, ji)
        s[D].compute_inline()
        s[C].compute_at(s[E], jo)

        return s

    return compute, schedule


@exe
def insert_transpose():
    def compute():
        A = te.placeholder((128, 128), name='A')
        B = te.compute((128, 128), lambda i, j: A[j][i] + 1, name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        A_read = s.cache_read(A, 'local', [B])
        s[A_read].reorder(s[A_read].op.axis[1], s[A_read].op.axis[0])
        A_write = s.cache_write(A_read, 'local')
        s[A_read].compute_inline()

        return s

    return compute, schedule


@exe
def reduction_parallel():
    def compute():
        A = te.placeholder((16, 1024), name='A')
        k = te.reduce_axis((0, 1024), name='k')
        B = te.compute((16,), lambda i: te.sum(A[i][k], axis=k), name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        k, = s[B].op.reduce_axis
        ko, ki = s[B].split(k, nparts=8)
        BF = s.rfactor(B, ko, factor_axis=1)

        i, ko = s[BF].op.axis
        s[BF].parallel(ko)

        return s

    return compute, schedule


@exe
def reduction_vectorize():
    def compute():
        A = te.placeholder((16, 1024), name='A')
        k = te.reduce_axis((0, 1024), name='k')
        B = te.compute((16,), lambda i: te.sum(A[i][k], axis=k), name='B')

        return [A, B]

    def schedule(bufs):
        A, B = bufs
        s = te.create_schedule([B.op])

        k, = s[B].op.reduce_axis
        ko, ki = s[B].split(k, 8)
        BF = s.rfactor(B, ki, factor_axis=1)

        i, ki = s[BF].op.axis
        s[BF].vectorize(ki)

        return s

    return compute, schedule


def gen_all(args):
    for i, exe in enumerate(EXE_REGISTRY):
        compute, schedule = exe()

        # Print compute definition
        print("-" * 60)
        print("-" * 15 + " Ex.%2d : %-20s " % (i+1, exe.__name__) + "-" * 15)
        print("-" * 60)
        print("=" * 10 + " Compute " + "=" * 10)
        print(pretty_print_func(compute))
        bufs = compute()

        # Print input IR
        s0 = te.create_schedule([bufs[-1].op])
        print("=" * 10 + " Input IR " + "=" * 10)
        print(pretty_print_ir(tvm.lower(s0, bufs, simple_mode=True)))

        # Print output IR
        s1 = schedule(bufs)
        print("=" * 10 + " Output IR " + "=" * 10)
        print(pretty_print_ir(tvm.lower(s1, bufs, simple_mode=True)))

        if args.solution:
            # Print reference schedule
            print("=" * 10 + " Reference Schedule " + "=" * 10)
            print(pretty_print_func(schedule))

        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution', action='store_true')
    args = parser.parse_args()

    gen_all(args)

