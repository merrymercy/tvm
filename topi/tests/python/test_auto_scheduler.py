import tvm
import topi
from tvm import autotvm

target_list = [tvm.target.cuda(), tvm.target.create('llvm')]

def test_reduction():
    def pool():
        A = tvm.placeholder((1, 256, 7, 7), name='A')
        D = topi.nn.pool(A, (2, 2), (1, 1), (0, 0, 0, 0), 'max')
        D = topi.nn.relu(D)
        return A, D

    def global_pool():
        A = tvm.placeholder((1, 256, 7, 7), name='A')
        D = topi.nn.global_pool(A, 'max')
        D = topi.nn.relu(D)
        return A, D

    def norm3():
        D1 = D2 = D3 = 128

        A = tvm.placeholder((D1, D2, D3), name='A')

        r1 = tvm.reduce_axis((0, D1), name='r1')
        r2 = tvm.reduce_axis((0, D2), name='r2')
        r3 = tvm.reduce_axis((0, D3), name='r3')

        D = tvm.compute((1,), lambda i: tvm.sum(A[r1, r2, r3] * A[r1, r2, r3], axis=[r1, r2, r3]), name='C')
        D = topi.nn.relu(D)
        return A, D

    def argmin():
        A = tvm.placeholder((1, 200, 20), name='A')
        D = topi.argmin(A, axis=(1, 2))
        D = topi.nn.relu(D)
        return A, D

    def softmax():
        A = tvm.placeholder((1, 20, 1000), name='A')
        D = topi.nn.softmax(A, axis=1)
        D = topi.nn.relu(D)
        return A, D

    test_cast_list = [
        pool(), global_pool(), norm3(),  softmax(), argmin(),
    ]

    for target in target_list:
        with target:
            for buffers in test_cast_list:
                autotvm.create_schedule(buffers)

if __name__ == "__main__":
    test_reduction()
