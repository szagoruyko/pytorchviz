import unittest
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace


def make_mlp_and_input():
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))
    x = torch.randn(1, 8)
    return model, x


class TestTorchviz(unittest.TestCase):

    def test_mlp_make_dot(self):
        model, x = make_mlp_and_input()
        y = model(x)
        dot = make_dot(y.mean(), params=dict(model.named_parameters()))

    def test_mlp_make_dot_from_trace(self):
        model, x = make_mlp_and_input()
        with torch.onnx.set_training(model, False):
            trace, _ = torch.jit.get_trace_graph(model, args=(x,))
        dot = make_dot_from_trace(trace)

    def test_double_backprop_make_dot(self):
        model, x = make_mlp_and_input()
        x.requires_grad = True

        def double_backprop(inputs, net):
            y = net(x).mean()
            grad, = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)
            return grad.pow(2).mean() + y

        dot = make_dot(double_backprop(x, model), params=dict(list(model.named_parameters()) + [('x', x)]))

    def test_lstm_make_dot(self):
        lstm_cell = nn.LSTMCell(128, 128)
        x = torch.randn(1, 128)
        dot = make_dot(lstm_cell(x), params=dict(list(lstm_cell.named_parameters())))


if __name__ == '__main__':
    unittest.main()
