
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from .common import no_dropout, no_layer_norm, get_indicator, get_module_device


class RNNFrame(nn.Module):
    def __init__(self, rnn_cells, for_lstm=False, batch_first=False, dropout=0, bidirectional=False):
        """
        :param rnn_cells: ex) [(cell_0_f, cell_0_b), (cell_1_f, cell_1_b), ..]
        :param dropout:
        :param bidirectional:
        """
        super().__init__()

        if bidirectional:
            assert all(len(pair) == 2 for pair in rnn_cells)
        elif not any(isinstance(rnn_cells[0], iterable)
                     for iterable in [list, tuple, nn.ModuleList]):
            rnn_cells = tuple((cell,) for cell in rnn_cells)

        self.rnn_cells = nn.ModuleList(nn.ModuleList(pair)
                                       for pair in rnn_cells)
        self.for_lstm = for_lstm
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = len(rnn_cells)

        if dropout > 0 and self.num_layers > 1:
            # dropout is applied to output of each layer except the last layer
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = no_dropout

        self.batch_first = batch_first

    def get_zero_init_state(self, hidden_size):
        # init_state with heterogenous hidden_size
        if self.for_lstm:
            init_hidden = init_cell = [
                torch.zeros(hidden_size,
                            self.rnn_cells[layer_idx][direction].hidden_size,
                            device=get_module_device(self))
                for layer_idx in range(self.num_layers)
                for direction in range(self.num_directions)]
            init_state = init_hidden, init_cell
        else:
            init_state = [
                torch.zeros(hidden_size,
                            self.rnn_cells[layer_idx][direction].hidden_size,
                            device=get_module_device(self))
                for layer_idx in range(self.num_layers)
                for direction in range(self.num_directions)]
        return init_state

    def get_init_step_state(self, init_state, state_idx):
        if self.for_lstm:
            init_hidden, init_cell = init_state
            step_state = (init_hidden[state_idx], init_cell[state_idx])
        else:
            step_state = init_state[state_idx]
        return step_state

    def get_step_output(self, step_state):
        if self.for_lstm:
            h, c = step_state
            step_output = h
        else:
            step_output = step_state
            
        return step_output

    def get_direction_last_state(self, step_state_list, lengths):
        if self.for_lstm:
            direction_last_state = tuple(
                torch.stack([h_or_c[length - 1][example_id]
                             for example_id, length in enumerate(lengths)], dim=0)
                for h_or_c in zip(*step_state_list))
            # direction_last_hidden, direction_last_cell = direction_last_state
        else:
            direction_last_state = \
                torch.stack([step_state_list[length - 1][example_id]
                             for example_id, length in enumerate(lengths)], dim=0)
        return direction_last_state

    def get_last_state(self, direction_last_state_list):
        if self.for_lstm:
            last_state = tuple(
                torch.stack(direction_last_h_or_c_list, dim=0)
                for direction_last_h_or_c_list in zip(*direction_last_state_list))
            # h_n, c_n = last_state
        else:
            last_state = torch.stack(direction_last_state_list, dim=0)
        return last_state

    def align_sequence(self, seq, lengths, shift_right):
        """
        :param seq: (seq_len, batch_size, *)
        """
        multiplier = 1 if shift_right else -1
        example_seqs = torch.split(seq, 1, dim=1)
        max_length = max(lengths)
        shifted_seqs = [example_seq.roll((max_length - length) * multiplier, dims=0)
                        for example_seq, length in zip(example_seqs, lengths)]
        return torch.cat(shifted_seqs, dim=1)

    def forward(self, input, init_state=None):
        """
        :param input: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: (h_0, c_0) where the size of both is (num_layers * num_directions, batch, hidden_size)
        :returns: (output, (h_n, c_n))
        - output: (seq_len, batch, num_directions * hidden_size)
        - h_n: (num_layers * num_directions, batch, hidden_size)
        - c_n: (num_layers * num_directions, batch, hidden_size)
        """

        if isinstance(input, torch.nn.utils.rnn.PackedSequence):
            input_packed = True
            # always batch_first=False --> trick to process input regardless of batch_first option
            input, lengths = pad_packed_sequence(input, batch_first=False)
            if max(lengths) == min(lengths):
                uniform_length = True
            else:
                uniform_length = False
            if isinstance(lengths, torch.Tensor):
                lengths = tuple(lengths.detach().cpu().numpy())
            assert max(lengths) == input.size()[0]
        else:
            input_packed = False
            if self.batch_first:
                input = input.transpose(0, 1)
            lengths = [input.size()[0]] * input.size()[1]
            uniform_length = True

        if not uniform_length:
            indicator = get_indicator(torch.tensor(lengths, device=get_module_device(self)))

        if init_state is None:
            init_state = self.get_zero_init_state(input.size()[1])

        direction_last_state_list = []
        layer_output = input

        for layer_idx in range(self.num_layers):
            layer_input = layer_output
            if layer_idx != 0:
                layer_input = self.dropout(layer_input)

            direction_output_list = []

            for direction in range(self.num_directions):
                cell = self.rnn_cells[layer_idx][direction]
                state_idx = layer_idx * self.num_directions + direction
                step_state = self.get_init_step_state(init_state, state_idx)

                direction_output = torch.zeros(
                    layer_input.size()[:2] + (cell.hidden_size,),
                    device=get_module_device(self))  # (seq_len, batch_size, hidden_size)
                step_state_list = []

                if direction == 0:
                    step_input_gen = enumerate(layer_input)
                else:
                    step_input_gen = reversed(list(enumerate(
                        layer_input if uniform_length else
                        self.align_sequence(layer_input, lengths, True))))

                for seq_idx, cell_input in step_input_gen:
                    step_state = cell(cell_input, step_state)
                    direction_output[seq_idx] = self.get_step_output(step_state)
                    step_state_list.append(step_state)
                if direction == 1 and not uniform_length:
                    direction_output = self.align_sequence(
                        direction_output, lengths, False)

                if uniform_length:
                    # hidden & cell's size = (batch, hidden_size)
                    direction_last_state = step_state_list[-1]
                else:
                    direction_last_state = self.get_direction_last_state(step_state_list, lengths)

                direction_output_list.append(direction_output)
                direction_last_state_list.append(direction_last_state)

            if self.num_directions == 2:
                assert direction_output_list[0].size() == direction_output_list[1].size()
                layer_output = torch.stack(direction_output_list, dim=2).view(
                    direction_output_list[0].size()[:2] + (-1,))
            else:
                layer_output = direction_output_list[0]

        output = layer_output
        last_state = self.get_last_state(direction_last_state_list)

        if not uniform_length:
            # the below one line code cleans out trash values beyond the range of lengths.
            # actually, the code is for debugging, so it can be removed to enhance computing speed slightly.
            output = (
                output.transpose(0, 1) * indicator).transpose(0, 1)

        if input_packed:
            output = pack_padded_sequence(output, lengths, batch_first=self.batch_first)
        elif self.batch_first:
            output = output.transpose(0, 1)

        return output, last_state


class LSTMFrame(RNNFrame):
    "Wrapper of RNNFrame. The 'for_lstm' option is always 'True'."

    def __init__(self, rnn_cells, batch_first=False, dropout=0, bidirectional=False):
        super().__init__(rnn_cells,
                         for_lstm=True,
                         batch_first=batch_first,
                         dropout=dropout,
                         bidirectional=bidirectional)


class LSTMCell(nn.Module):
    """
    standard LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fiou_linear = nn.Linear(input_size + hidden_size, hidden_size * 4)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a pair of a hidden tensor and a cell tensor whose shape is (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: 1-dimensional hidden and cell
        """
        hidden_tensor, cell_tensor = state

        fio_linear, u_linear = torch.split(
            self.fiou_linear(torch.cat([input, hidden_tensor], dim=1)),
            self.hidden_size * 3, dim=1)

        f, i, o = torch.split(torch.sigmoid(fio_linear),
                              self.hidden_size, dim=1)
        u = torch.tanh(u_linear)

        new_cell = i * u + (f * cell_tensor)
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, layer_norm_enabled=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            input_size + hidden_size, hidden_size, bias=not layer_norm_enabled)

        # if dropout is not None:
        #     if isinstance(dropout, nn.Dropout):
        #         self.dropout = dropout
        #     elif dropout > 0:
        #         self.dropout = nn.Dropout(dropout)
        #     else:
        #         self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = no_layer_norm

    def forward(self, input, hidden):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a hidden tensor of shape (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: hidden and cell
        """
        return torch.tanh(self.layer_norm(self.linear(
            torch.cat([input, hidden], dim=1))))


class LayerNormLSTMCell(nn.Module):
    """
    It's based on tf.contrib.rnn.LayerNormBasicLSTMCell
    Reference:
    - https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell
    - https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1335
    """

    def __init__(self, input_size, hidden_size, dropout=None, layer_norm_enabled=True, cell_ln=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fiou_linear = nn.Linear(
            input_size + hidden_size, hidden_size * 4, bias=not layer_norm_enabled)

        if dropout is not None:
            # recurrent dropout is applied
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                assert dropout >= 0
                self.dropout = no_dropout
        else:
            self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.fiou_ln_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(4))
            # self.fiou_ln_layers = nn.ModuleList(
            #     nn.LayerNorm(hidden_size) for _ in range(3))
            # self.fiou_ln_layers.append(
            #     nn.LayerNorm(hidden_size) if u_ln is None else u_ln)
            self.cell_ln = nn.LayerNorm(
                hidden_size) if cell_ln is None else cell_ln
        else:
            assert cell_ln is None
            # assert u_ln is cell_ln is None
            self.fiou_ln_layers = (no_layer_norm,) * 4
            self.cell_ln = no_layer_norm
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a pair of a hidden tensor and a cell tensor whose shape is (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: hidden and cell
        """
        hidden_tensor, cell_tensor = state

        fiou_linear = self.fiou_linear(
            torch.cat([input, hidden_tensor], dim=1))
        fiou_linear_tensors = fiou_linear.split(self.hidden_size, dim=1)

        # if self.layer_norm_enabled:
        fiou_linear_tensors = tuple(ln(tensor) for ln, tensor in zip(
            self.fiou_ln_layers, fiou_linear_tensors))

        f, i, o = tuple(torch.sigmoid(tensor)
                        for tensor in fiou_linear_tensors[:3])
        u = self.dropout(torch.tanh(fiou_linear_tensors[3]))

        new_cell = self.cell_ln(i * u + (f * cell_tensor))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class LayerNormLSTM(LSTMFrame):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, r_dropout=0, bidirectional=False, layer_norm_enabled=True):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.r_dropout = r_dropout
        self.bidirectional = bidirectional
        self.layer_norm_enabled = layer_norm_enabled

        r_dropout_layer = nn.Dropout(r_dropout)
        rnn_cells = tuple(
            tuple(
                LayerNormLSTMCell(
                    input_size if layer_idx == 0 else hidden_size * (2 if bidirectional else 1),
                    hidden_size,
                    dropout=r_dropout_layer,
                    layer_norm_enabled=layer_norm_enabled)
                for _ in range(2 if bidirectional else 1))
            for layer_idx in range(num_layers))

        super().__init__(rnn_cells=rnn_cells, dropout=dropout,
                         batch_first=batch_first, bidirectional=bidirectional)


def forward_rnn(rnn, init_state, input, lengths, batch_first=False,
                embedding: torch.nn.Embedding = None,
                dropout: torch.nn.Dropout = None,
                return_packed_output=False):
    # "batch_first" means whether "input" is a batch-first tensor
    padded = pad_sequence(input, batch_first=batch_first)
    if embedding is not None:
        padded = embedding(padded)
    if dropout is not None:
        padded = dropout(padded)
    packed = pack_padded_sequence(padded, lengths, batch_first=batch_first, enforce_sorted=False)
    packed_output, last_state = rnn(packed, init_state)
    # (ht, ct) = last_state  # when rnn is a lstm
    if return_packed_output:
        return packed_output, last_state
    else:
        output, lengths2 = pad_packed_sequence(packed_output, batch_first=batch_first)
        return output, last_state
