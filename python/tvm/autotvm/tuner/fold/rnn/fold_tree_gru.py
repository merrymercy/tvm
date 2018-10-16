# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys

import mxnet as mx
from mxnet import nd
from mxnet.gluon import Block, nn, rnn
from mxnet.gluon.parameter import Parameter
from mxnet.gluon.rnn.rnn_cell import _format_sequence, _get_begin_state

class ChildSumGRUCell(rnn.HybridRecurrentCell):
    def __init__(self, hidden_size,
                 num_children,
                 i2h_weight_initializer=None,
                 h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros',
                 h2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None):
        super(ChildSumGRUCell, self).__init__(prefix=prefix, params=params)
        self.num_children = num_children
        with self.name_scope():
            self._hidden_size = hidden_size
            self._input_size = input_size
            self.i2h = nn.Dense(3*hidden_size,
                                weight_initializer=i2h_weight_initializer,
                                bias_initializer=i2h_bias_initializer)
            self.h2h = nn.Dense(3*hidden_size,
                                weight_initializer=h2h_weight_initializer,
                                bias_initializer=h2h_bias_initializer)

    def _alias(self):
        return 'childsum_gru'

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'}]

    def hybrid_forward(self, F, inputs, children_states):
        prefix = '{0}{1}_'.format(self.prefix, self._alias)
        children_hidden_states = children_states
        assert not self.num_children or len(children_hidden_states) == self.num_children

        prev_state_h = F.add_n(*children_hidden_states, name=prefix + 'hs')

        i2h = self.i2h(inputs)
        h2h = self.h2h(prev_state_h)

        i2h_r, i2h_z, i2h = F.SliceChannel(i2h, num_outputs=3, name=prefix + 'i2h_slice')
        h2h_r, h2h_z, h2h = F.SliceChannel(h2h, num_outputs=3, name=prefix + 'h2h_slice')

        reset_gate = F.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                  name=prefix + 'r_act')
        update_gate = F.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                   name=prefix + 'z_act')

        next_h_tmp = F.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                  name=prefix + 'h_act')
        next_h = F._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                   name=prefix + 'out')

        return next_h, [next_h]

    @staticmethod
    def batch_forward(idx, children, inputs, to_slots, num_slots, mem_slots, cells):
        """forward a batch input if their tree structures are the same"""
        root_input = inputs[:, idx:idx + 1, :]
        n_children = children[idx][0]
        batch_size = inputs.shape[0]

        if n_children:
            root_h = [ChildSumGRUCell.batch_forward(
                          c, children, inputs, to_slots, num_slots, mem_slots, cells)[1][0]
                      for c in children[idx][1:1 + n_children]]
        else:
            with root_input.context:
                root_h = [cells[0].begin_state(len(inputs))[0]]
        root_input = root_input.reshape((batch_size, -1))
        ret = cells[n_children](root_input, root_h)
        out = ret[0]

        if mem_slots is not None and n_children:
            weight = to_slots(out)
            weight = mx.nd.softmax(weight)
            weight = weight.reshape((batch_size, num_slots, 1))
            out = mx.nd.tile(out, (1, num_slots)).reshape((batch_size, num_slots, -1))
            slots = mx.nd.broadcast_mul(out, weight).reshape((batch_size, -1))
            mem_slots.append(slots)
        return ret

    @staticmethod
    def encode(cells, inputs, tree):
        root_input = inputs[tree.idx:(tree.idx+1)]
        if tree.children:
            root_h = [ChildSumGRUCell.encode(cells, inputs, c)[1][0] for c in tree.children]
        else:
            with root_input.context:
                root_h = [cells[0].begin_state(1)[0]]
        num_children = len(tree.children)
        return cells[num_children](root_input, root_h)

    @staticmethod
    def cell_forward(cell, inputs, states):
        return cell(inputs, states)

    @staticmethod
    def fold_encode(fold, cells, inputs, tree):
        root_input = inputs[tree.idx:(tree.idx+1)]
        if tree.children:
            root_h = [ChildSumGRUCell.fold_encode(fold, cells, inputs, c)[1][0]
                      for c in tree.children]
        else:
            root_h = [fold.record(0, cells[0].begin_state, 1).no_batch()[0]]
        num_children = len(tree.children)
        return fold.record(0, ChildSumGRUCell.cell_forward, cells[num_children], root_input, root_h)
