import itertools
import time
import os

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist

import tvm
from tvm import autotvm
import topi

batch_size = 32
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def batches(x, y):
    for i in range(int(x.shape[0] / batch_size)):
        yield (x[i:i+batch_size, :, :, None].astype('float32'),
               y[i:i+batch_size, ...].astype('float32'))

def make_keras_model():
    data = keras.layers.Input(shape=(28, 28, 1))
    x = data
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    keras_model = keras.models.Model(data, x)

    keras_model.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'],
                        optimizer=keras.optimizers.SGD(lr=1e-2))
    
    return keras_model

weights = []

x = tvm.placeholder((batch_size, 28, 28, 1))
y = tvm.placeholder((batch_size, num_classes))

t = topi.transpose(x, [0, 3, 1, 2])

w1 = tvm.placeholder((32, 1, 3, 3), name="w1")
b1 = tvm.placeholder((32,), name="b1")
t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0, 1) + topi.reshape(b1, (1, 32, 1, 1)))
weights.extend([w1, b1])

w2 = tvm.placeholder((64, 32, 3, 3), name="w2")
b2 = tvm.placeholder((64,), name="b2")
t = topi.nn.relu(topi.nn.conv2d(t, w2, 1, 0, 1) + topi.reshape(b2, (1, 64, 1, 1)))
weights.extend([w2, b2])

t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'max')

# Note that we have to transpose before flatten
t = topi.transpose(t, [0, 2, 3, 1])
t = topi.nn.flatten(t)

w3 = tvm.placeholder((128, np.prod([s.value for s in t.shape[1:]])), name="w3")
b3 = tvm.placeholder((128,), name="b3")
t = topi.nn.relu(topi.nn.dense(t, w3, b3))
weights.extend([w3, b3])

w4 = tvm.placeholder((num_classes, 128), name="w4")
b4 = tvm.placeholder((num_classes,), name="b4")
t = topi.nn.dense(t, w4, b4)
weights.extend([w4, b4])

# We use a custom softmax because the topi implementation uses max behind the scenes
# which currently causes problems with autodiff, leading to poor performance
exps = topi.exp(t)
sumexps = topi.sum(exps, axis=-1, keepdims=True)
logsoftmax = topi.log(exps/sumexps)

loss = - topi.sum(y * logsoftmax) / batch_size

gradients = list(tvm.differentiate(loss, weights))
learning_rate = tvm.placeholder(())
new_weights = [w - learning_rate*g for w, g in zip(weights, gradients)]

target = tvm.target.create('llvm')
ctx = tvm.context(str(target))

# get tensor's shape as a list
def get_shape(tensor):
    return [s.value for s in tensor.shape]

# empty tensor values for a list of tensors or just a single tensor
def empty_val(tensor):
    if isinstance(tensor, list):
        return [empty_val(t) for t in tensor]
    else:
        return tvm.nd.empty(get_shape(tensor), tensor.dtype, ctx=ctx)

def schedule_somehow(sched):
    # This autoinlining turned aoyt to be harmful because it prevents memoization of some expensive operations
    # tvm.schedule.AutoInlineInjective(sched)
    for s in sched.stages:
        if isinstance(s.op, tvm.tensor.ComputeOp) and isinstance(s.op.body[0], tvm.expr.Reduce):
            ax = s.fuse(*s.op.axis)
            axo, axi = s.split(ax, nparts=12)
            s.parallel(axo)

sched = tvm.create_schedule(loss.op)
schedule_somehow(sched)
testing_module = tvm.build(sched, [loss, x, y] + weights)
training_module = None

#with target:
#    bufs = [loss, x, y] + weights
#    s, bufs = autotvm.create_schedule(bufs)
#    testing_module = tvm.build(s, bufs)

#    bufs = [loss, x, y, learning_rate] + new_weights + weights
#    s, bufs = autotvm.create_schedule(bufs)
#    training_module = tvm.build(sched, bufs)
#    training_module = None

class TVMModel:
    def __init__(self):
        self.weights_values = empty_val(weights)
        
    def test(self, xval, yval):
        args = [empty_val(loss)] + [tvm.ndarray.array(xval, ctx=ctx), tvm.ndarray.array(yval, ctx=ctx)] + self.weights_values
        testing_module(*args)
        return args[0].asnumpy()

    def train(self, xval, yval, lr=1e-2):
        new_weights_values = empty_val(new_weights)
        args = [empty_val(loss)] + [tvm.ndarray.array(xval.astype('float32')), tvm.ndarray.array(yval.astype('float32'))] +                [tvm.ndarray.array(np.array(lr).astype('float32'))] + new_weights_values + self.weights_values
        training_module(*args)
        for wv, new_wv in zip(self.weights_values, new_weights_values):
            wv.copyfrom(new_wv)
        return args[0].asnumpy()
    
    def from_keras(self, keras_model):
        assert len(keras_model.get_weights()) == len(self.weights_values)
        for kv, wv in zip(keras_model.get_weights(), self.weights_values):
            if len(kv.shape) == 4:
                kv = np.transpose(kv, [3, 2, 0, 1])
            elif len(kv.shape) == 2:
                kv = np.transpose(kv)

            #print(wv.shape, " <- ", kv.shape)

            wv.copyfrom(kv)
            
    def to_keras(self, keras_model):
        assert len(keras_model.get_weights()) == len(self.weights_values)
        new_keras_weights = []
        for kv, wv in zip(keras_model.get_weights(), self.weights_values):
            wv_np = wv.asnumpy()
            if len(kv.shape) == 4:
                wv_np = np.transpose(wv_np, [2, 3, 1, 0])
            elif len(kv.shape) == 2:
                wv_np = np.transpose(wv_np)
            new_keras_weights.append(wv_np)

        keras_model.set_weights(new_keras_weights)

keras_model = make_keras_model()
keras_model.summary()

tvm_model = TVMModel()


tic = time.time()
for xx, yy in itertools.islice(batches(x_train, y_train), 1000):
    keras_model.test_on_batch(xx, yy)
print("TF: %.2f" % (time.time() - tic))

tic = time.time()
for xx, yy in itertools.islice(batches(x_train, y_train), 1000):
    tvm_model.test(xx, yy)
print("TVM: %.2f" % (time.time() - tic))

tic = time.time()
for xx, yy in itertools.islice(batches(x_train, y_train), 1000):
    keras_model.test_on_batch(xx, yy)
print("TF: %.2f" % (time.time() - tic))

tic = time.time()
for xx, yy in itertools.islice(batches(x_train, y_train), 1000):
    tvm_model.test(xx, yy)
print("TVM: %.2f" % (time.time() - tic))

