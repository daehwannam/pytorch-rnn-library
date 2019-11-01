# Copyright (c) 2019 Daehwan Nam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch
import torch.nn.functional as F

from .common import no_dropout, no_layer_norm


class ChildSumTreeLSTMCell(nn.Module):
    # https://arxiv.org/abs/1503.00075
    def __init__(self, input_size, hidden_size, learn_init_state=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.iou_linear = nn.Linear(input_size + hidden_size, hidden_size * 3)
        self.f_input_linear = nn.Linear(
            input_size, hidden_size, bias=False)
        self.f_hidden_linear = nn.Linear(hidden_size, hidden_size)

        self.init_hidden = nn.Parameter(torch.zeros(hidden_size), learn_init_state)
        self.init_cell = nn.Parameter(torch.zeros(hidden_size), learn_init_state)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self, input, states=None):
        """
        :param input: 1-dimensional tensor
        :param states: pairs of 1-dimensional hiddens and cells.
                       ex. ((h_0, c_0), (h_1, c_1), (h_2, c_2))
        :returns: 1-dimensional hidden and cell
        """
        # states: pairs of (cell, hidden)
        if states is None:
            states = ((self.init_hidden, self.init_cell),)

        num_children = len(states)
        hiddens, cells = zip(*states)
        hidden_tensor = torch.cat([hidden.view(1, self.hidden_size)
                                   for hidden in hiddens], dim=0)
        cell_tensor = torch.cat([cell.view(1, self.hidden_size)
                                 for cell in cells], dim=0)

        io_linear, u_linear = torch.split(
            self.iou_linear(torch.cat([input, hidden_tensor.sum(0)], dim=0)),
            self.hidden_size * 2, dim=0)

        i, o = torch.split(torch.sigmoid(io_linear), self.hidden_size, dim=0)
        u = torch.tanh(u_linear)

        f = torch.sigmoid(self.f_input_linear(input).repeat(num_children, 1) +
                          self.f_hidden_linear(hidden_tensor))

        new_cell = i * u + (f * cell_tensor).sum(0)
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class LayerNormChildSumTreeLSTMCell(nn.Module):
    """Combination of tree LSTM and layer normalization & recurrent dropout without memory loss"""

    def __init__(self, input_size, hidden_size, dropout=None, layer_norm_enabled=True, learn_init_state=False, cell_ln=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.iou_linear = nn.Linear(
            input_size + hidden_size, hidden_size * 3, bias=not layer_norm_enabled)
        self.f_input_linear = nn.Linear(
            input_size, hidden_size, bias=False)
        self.f_hidden_linear = nn.Linear(
            hidden_size, hidden_size, bias=not layer_norm_enabled)

        self.init_hidden = nn.Parameter(torch.zeros(hidden_size), learn_init_state)
        self.init_cell = nn.Parameter(torch.zeros(hidden_size), learn_init_state)

        if dropout is not None:
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.f_ln = nn.LayerNorm(hidden_size)
            self.iou_ln_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(3))
            # self.iou_ln_layers.append(
            #     nn.LayerNorm(hidden_size) if u_ln is None else u_ln)
            self.cell_ln = nn.LayerNorm(
                hidden_size) if cell_ln is None else cell_ln

        else:
            assert cell_ln is None
            # assert u_ln is cell_ln is None
            self.f_ln = no_layer_norm
            self.iou_ln_layers = (no_layer_norm,) * 3
            self.cell_ln = no_layer_norm

    def forward(self, input, states=None):
        """
        :param input: 1-dimensional tensor
        :param states: pairs of 1-dimensional hiddens and cells.
                       ex. ((h_0, c_0), (h_1, c_1), (h_2, c_2))
        :returns: 1-dimensional hidden and cell
        """
        # states: pairs of (cell, hidden)
        if states is None:
            states = ((self.init_hidden, self.init_cell),)

        # num_children = len(states)
        hiddens, cells = zip(*states)
        hidden_tensor = torch.stack(hiddens, dim=0)
        cell_tensor = torch.stack(cells, dim=0)

        f_linear_tensor = self.f_input_linear(input) + \
            self.f_hidden_linear(hidden_tensor)
        # f_linear_tensor = self.f_input_linear(input).repeat(num_children, 1) + \
        #     self.f_hidden_linear(hidden_tensor)

        iou_linear_tensors = torch.split(
            self.iou_linear(torch.cat([input, hidden_tensor.sum(0)], dim=0)),
            self.hidden_size, dim=0)

        # if self.layer_norm_enabled:
        f_linear_tensor = torch.stack(
            [self.f_ln(tensor) for tensor in f_linear_tensor],
            dim=0)
        iou_linear_tensors = tuple(ln(tensor) for ln, tensor in
                                   zip(self.iou_ln_layers, iou_linear_tensors))

        f = torch.sigmoid(f_linear_tensor)
        i, o = (torch.sigmoid(tensor) for tensor in iou_linear_tensors[:2])
        u = self.dropout(torch.tanh(iou_linear_tensors[2]))

        new_cell = self.cell_ln(i * u + (f * cell_tensor).sum(0))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class AttentiveChildSumTreeLSTMCell(nn.Module):
    """
    The model is based on "Modelling Sentence Pairs with Tree-structured Attentive Encoder, COLING 2016"
    And, layer normalization & recurrent dropout without memory loss is applied.
    (Also, elementwise attention is added as an option.)
    """

    def __init__(self, input_size, hidden_size, external_size=None, dropout=None,
                 attention_enabled=True, elementwise=False, layer_norm_enabled=True,
                 learn_init_state=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if attention_enabled:
            assert external_size
            self.external_size = external_size

            self.attention_input_linear = nn.Linear(
                hidden_size +
                external_size, (hidden_size + external_size) // 2,
                bias=False)
            self.attention_linear = nn.Linear(
                (hidden_size + external_size) // 2, hidden_size if elementwise else 1,
                bias=False)
            self.merge_linear = nn.Linear(
                hidden_size, hidden_size, bias=not layer_norm_enabled)

            if layer_norm_enabled:
                self.merge_ln = nn.LayerNorm(hidden_size)
            else:
                self.merge_ln = no_layer_norm
            self.forward_merge = self.forward_attentive_merge
        else:
            self.forward_merge = self.forward_sum_merge

        self.iou_linear = nn.Linear(
            input_size + hidden_size, hidden_size * 3, bias=not layer_norm_enabled)
        self.f_input_linear = nn.Linear(
            input_size, hidden_size, bias=False)
        self.f_hidden_linear = nn.Linear(
            hidden_size, hidden_size, bias=not layer_norm_enabled)

        self.init_hidden = nn.Parameter(torch.zeros(hidden_size), learn_init_state)
        self.init_cell = nn.Parameter(torch.zeros(hidden_size), learn_init_state)

        if dropout is not None:
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.f_ln = nn.LayerNorm(hidden_size)
            self.iou_ln_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(3))
            self.cell_ln = nn.LayerNorm(hidden_size)
        else:
            self.f_ln = no_layer_norm
            self.iou_ln_layers = (no_layer_norm,) * 3
            self.cell_ln = no_layer_norm

    def forward(self, input, states=None, external=None):
        """
        :param input: 1-dimensional tensor
        :param states: pairs of 1-dimensional hiddens and cells.
                       ex. ((h_0, c_0), (h_1, c_1), (h_2, c_2))
        :external: 1-dimensional tensor
        :returns: 1-dimensional hidden and cell
        """
        if states is None:
            states = ((self.init_hidden, self.init_cell),)
            # assert not external
            # external = enable_cuda(self, torch.zeros(self.hidden_size))

        hiddens, cells = zip(*states)
        hidden_tensor = torch.stack(hiddens, dim=0)
        cell_tensor = torch.stack(cells, dim=0)

        f_linear_tensor = self.f_input_linear(input) + \
            self.f_hidden_linear(hidden_tensor)

        iou_linear_tensors = torch.split(
            self.iou_linear(torch.cat(
                [input, self.forward_merge(hidden_tensor, external)], dim=0)),
            self.hidden_size, dim=0)

        f_linear_tensor = torch.stack(
            [self.f_ln(tensor) for tensor in f_linear_tensor],
            dim=0)
        iou_linear_tensors = tuple(ln(tensor) for ln, tensor in
                                   zip(self.iou_ln_layers, iou_linear_tensors))

        f = torch.sigmoid(f_linear_tensor)
        i, o = (torch.sigmoid(tensor) for tensor in iou_linear_tensors[:2])
        u = self.dropout(torch.tanh(iou_linear_tensors[2]))

        new_cell = self.cell_ln(i * u + (f * cell_tensor).sum(0))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell

    def forward_sum_merge(self, hidden_tensor, external):
        return hidden_tensor.sum(0)

    def forward_attentive_merge(self, hidden_tensor, external):
        # if external is None:
        #     return self.forward_sum_merge(hidden_tensor, external)

        attention_input = torch.tanh(self.attention_input_linear(
            torch.cat([hidden_tensor, external.repeat(hidden_tensor.size(0), 1)], dim=1)))
        attention = F.softmax(
            self.attention_linear(attention_input),
            dim=0)
        merge_hidden = torch.tanh(self.merge_ln(self.merge_linear(
            (hidden_tensor * attention).sum(0))))

        return merge_hidden


class LayerNormTreeGRUCell(nn.Module):
    """
    The model is based on "Modelling Sentence Pairs with Tree-structured Attentive Encoder, COLING 2016"
    And, layer normalization & recurrent dropout without memory loss is applied.

    code reference: https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940
    """

    def __init__(self, input_size, hidden_size,
                 dropout=None, layer_norm_enabled=True, init_hidden_enabled=False, learn_init_state=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.z_linear = nn.Linear(
            input_size + hidden_size, hidden_size, bias=not layer_norm_enabled)

        self.r_input_linear = nn.Linear(
            input_size, hidden_size, bias=False)
        self.r_hidden_linear = nn.Linear(
            hidden_size, hidden_size, bias=not layer_norm_enabled)

        self.u_linear = nn.Linear(
            input_size + hidden_size, hidden_size, bias=not layer_norm_enabled)

        if dropout is not None:
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.z_ln = nn.LayerNorm(hidden_size)
            self.r_ln = nn.LayerNorm(hidden_size)
            self.u_ln = nn.LayerNorm(hidden_size)
        else:
            self.z_ln = no_layer_norm
            self.r_ln = no_layer_norm
            self.u_ln = no_layer_norm

        self.init_hidden_enabled = init_hidden_enabled
        if init_hidden_enabled:
            self.init_hidden = nn.Parameter(torch.zeros(hidden_size), learn_init_state)

    def forward(self, input, hiddens=None):
        """
        :param input: 1-dimensional tensor
        :param states: 1-dimensional hiddens.
                       ex. (h_0, h_1, h_2)
        :returns: 1-dimensional hidden
        """
        # states: pairs of (cell, hidden)
        if hiddens is None:
            hiddens = (self.init_hidden,)

        # num_children = len(states)
        hidden_tensor = torch.stack(hiddens, dim=0)
        hidden_sum = hidden_tensor.sum(0)

        r = torch.sigmoid(torch.stack(
            [self.r_ln(tensor) for tensor in (
                self.r_input_linear(input) + self.r_hidden_linear(hidden_tensor))],
            dim=0))

        z = torch.sigmoid(self.z_ln(self.z_linear(
            torch.cat([input, hidden_sum], dim=0))))

        u = torch.tanh(self.u_ln(self.u_linear(
            torch.cat([input, (r * hidden_tensor).sum(0)], dim=0))))

        h = z * hidden_sum + (1 - z) * u

        return h


class LayerNormNoInputBinaryTreeLSTMCell(nn.Module):
    def __init__(self, hidden_size, dropout=None, layer_norm_enabled=True, learn_init_state=False, cell_ln=None):
        super().__init__()
        self.hidden_size = hidden_size

        self.ffiou_linear = nn.Linear(
            hidden_size * 2, hidden_size * 5, bias=not layer_norm_enabled)

        if dropout is not None:
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.ffio_ln_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(4))
            self.u_ln = nn.LayerNorm(hidden_size)
            # self.u_ln = nn.LayerNorm(
            #     hidden_size) if u_ln is None else u_ln
            self.cell_ln = nn.LayerNorm(
                hidden_size) if cell_ln is None else cell_ln
        else:
            # assert u_ln is cell_ln is None
            assert cell_ln is None
            self.ffio_ln_layers = (no_layer_norm,) * 4
            self.u_ln = no_layer_norm
            self.cell_ln = no_layer_norm

    def forward(self, state0, state1):
        """
        :param state0: (h0, c0); hidden and cell are 1-dimensional tensors
        :param state1: (h1, c1)
        :returns: (h_new, c_new)
        """

        hiddens, cells = zip(state0, state1)

        ffiou_linear_tensors = torch.split(
            self.ffiou_linear(torch.cat(hiddens, dim=0)), self.hidden_size, dim=0)

        f0, f1, i, o = (torch.sigmoid(ln(tensor)) for ln, tensor in
                        zip(self.ffio_ln_layers, ffiou_linear_tensors[:4]))
        u = self.dropout(torch.tanh(self.u_ln(ffiou_linear_tensors[4])))

        new_cell = self.cell_ln(i * u + f0 * cells[0] + f1 * cells[1])
        new_hidden = o * torch.tanh(new_cell)

        return new_hidden, new_cell


class LayerNormInputBinaryTreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=None, layer_norm_enabled=True, learn_init_state=False, cell_ln=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_linear = nn.Linear(input_size, hidden_size * 4)  # uiof
        self.lhidden_linear = nn.Linear(hidden_size, hidden_size * 5,
                                        bias=not layer_norm_enabled)  # uioff
        self.rhidden_linear = nn.Linear(hidden_size, hidden_size * 5)  # uioff

        if dropout is not None:
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.u_ln = nn.LayerNorm(hidden_size)
            self.ioff_ln_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(4))
            self.cell_ln = nn.LayerNorm(
                hidden_size) if cell_ln is None else cell_ln
        else:
            assert cell_ln is None
            self.u_ln = no_layer_norm
            self.ioff_ln_layers = (no_layer_norm,) * 4
            self.cell_ln = no_layer_norm

    def forward_sequential(self, input, state):
        # state == lstate
        hidden, cell = state

        input_linear = self.input_linear(input)
        hidden_linear = self.lhidden_linear(hidden)
        # last hidden (for right forget) is not used, so delete it
        hidden_linear = hidden_linear[:-self.hidden_size]
        uiof_linear_tensors = (
            input_linear + hidden_linear).split(self.hidden_size, dim=0)

        u = self.dropout(torch.tanh(self.u_ln(uiof_linear_tensors[0])))
        i, o, f = (torch.sigmoid(ln(tensor)) for ln, tensor in zip(
            self.ioff_ln_layers[:-1], uiof_linear_tensors[1:]))

        new_cell = self.cell_ln(i * u + (f * cell))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell

    def forward_binary(self, lstate, rstate):
        lhidden, lcell = lstate
        rhidden, rcell = rstate

        lhidden_linear = self.lhidden_linear(lhidden)
        rhidden_linear = self.rhidden_linear(rhidden)
        uioff_linear_tensors = (
            lhidden_linear + rhidden_linear).split(self.hidden_size, dim=0)

        u = self.dropout(torch.tanh(self.u_ln(uioff_linear_tensors[0])))
        i, o, lf, rf = (torch.sigmoid(ln(tensor)) for ln, tensor in zip(
            self.ioff_ln_layers, uioff_linear_tensors[1:]))

        new_cell = self.cell_ln(i * u + lf * lcell + rf * rcell)
        new_hidden = o * torch.tanh(new_cell)

        return new_hidden, new_cell
