import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend.interpreter import Value, TupleValue
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay import testing, create_executor, ir_pass

def test_simple_op():
    """
    fn elewise_add(A: tensor[(10, 10), 'float32'],
                   B: tensor[(10, 10), 'float32']) -> tensor[(10, 10), 'float32'] {
        compute(fn (i, j) { A[i, j] + B[i, j] * 2 }, [A, B], shape=[10, 10])
    }
    """
    A = relay.var('A', shape=(10, 10))
    B = relay.var('B', shape=(10, 10))

    i = relay.var('i', shape=(), dtype='int32')
    j = relay.var('j', shape=(), dtype='int32')

    expr = relay.Function([i, j], relay.add(relay.Index(A, [i,j]), 
                                            relay.Index(B, [i,j]) * relay.const(2.0)))
    z = relay.Function([A, B], relay.op._make.compute(expr, [10, 10], []))

    z = ir_pass.infer_type(z)
    z = ir_pass.parse_compute(z)

    target = 'llvm'
    ctx = tvm.context(target)
    A_np = np.random.randn(10, 10).astype('float32')
    B_np = np.random.randn(10, 10).astype('float32')

    intrp = create_executor(ctx=ctx, target=target)
    result = intrp.evaluate(z)(A_np, B_np)

    np.testing.assert_allclose(A_np + B_np * 2, result.asnumpy())

def test_simple_op_parser():
    text = """v0.0.1
    fn (%A: Tensor[(10, 10), float32], %B: Tensor[(10, 10), float32]) {
        let %C = compute(fn (%i, %j) {%A[%i, %j] + %B[%i, %j] * 2.0},
                         shape=(10, 10));

        compute(fn (%i, %j) { %C[%i, %j] + %A[%i / 2, %j / 2]},
                shape=(10, 10))
    }
    """
    z = relay.fromtext(text)

    z = ir_pass.infer_type(z)
    z = ir_pass.parse_compute(z)

    target = 'llvm'
    ctx = tvm.context(target)
    A_np = np.random.randn(10, 10).astype('float32')
    B_np = np.random.randn(10, 10).astype('float32')


    intrp = create_executor(ctx=ctx, target=target)
    result = intrp.evaluate(z)(A_np, B_np)

    C_np = A_np + B_np * 2.0
    for i in range(10):
        for j in range(10):
            C_np[i,j] = C_np[i,j] + A_np[i//2, j//2]

    np.testing.assert_allclose(C_np, result.asnumpy())

def test_simple_op_int():
    text = """v0.0.1
    fn () {
        let %expr = fn (%i:int32, %j:int32) { %i + %j };
        compute(%expr, shape=(10, 10))
    }
    """
    z = relay.fromtext(text)

    z = ir_pass.infer_type(z)
    z = ir_pass.parse_compute(z)

    target = 'llvm'
    ctx = tvm.context(target)
    intrp = create_executor(ctx=ctx, target=target)
    result = intrp.evaluate(z)().asnumpy()

    for i in range(10):
        for j in range(10):
            assert result[i][j] == i + j


def test_matmul():
    text = """v0.0.1
    fn (%A: Tensor[(10, 10), float32], %B: Tensor[(10, 10), float32]) {
        compute(fn (%i, %j, %k) { SUM(%A[%i, %k] * %B[%k, %j]) },
                shape=(10, 10), reduction=(10,))
    }
    """
    z = relay.fromtext(text)
    z = ir_pass.infer_type(z)

    z = ir_pass.parse_compute(z)

    target = 'llvm'
    ctx = tvm.context(target)
    A_np = np.random.randn(10, 10).astype('float32')
    B_np = np.random.randn(10, 10).astype('float32')

    intrp = create_executor(ctx=ctx, target=target)
    result = intrp.evaluate(z)(A_np, B_np)

    np.testing.assert_allclose(A_np.dot(B_np), result.asnumpy(), rtol=1e-2)

#def test_simple_op_grad():
#    text = """v0.0.1
#    fn (%A: Tensor[(10, 10), float32]) {
#        (fn (%C: (Tensor[(10, 10), float32], Tensor[(10, 10), float32])) {%C} )((%A, %A))
#    }
#    """
#    z = relay.fromtext(text)
#
#    print(z)
#    z = ir_pass.infer_type(z)
#    #z = ir_pass.parse_compute(z)
#
#    z = ir_pass.gradient(z)
#    print(z)
#
#    #target = 'llvm'
#    #ctx = tvm.context(target)
#    #A_np, B_np = np.ones((10, 10), dtype='float32'), np.ones((10, 10), dtype='float32')
#
#    #intrp = create_executor(ctx=ctx, target=target)
#    #result = intrp.evaluate(z)(A_np, B_np)
#
#    #np.testing.assert_allclose(A_np + B_np * 2, result.asnumpy())

if __name__ == "__main__":
    test_simple_op()
    test_simple_op_parser()
    test_simple_op_int()
    test_matmul()

