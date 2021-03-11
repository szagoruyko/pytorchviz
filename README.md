PyTorchViz
=======

A small package to create visualizations of PyTorch execution graphs and traces.

## Installation

Install graphviz, e.g.:

```
brew install graphviz
```

Install the package itself:

```
pip install torchviz
```


## Usage
Example usage of `make_dot`:
```
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

make_dot(y.mean(), params=dict(model.named_parameters()))
```
![image](https://user-images.githubusercontent.com/13428986/110844921-ff3f7500-8277-11eb-912e-3ba03623fdf5.png)

Set `show_attrs=True` and `show_saved=True` to see what autograd saves for the backward pass. (Note that this is only available for pytorch >= 1.9.)
```
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
```
![image](https://user-images.githubusercontent.com/13428986/110845186-4ded0f00-8278-11eb-88d2-cc33413bb261.png)

There are two functions, `make_dot` to make graphs from any PyTorch functions (requires that at least one input Variable requires_grad), and `make_dot_from_trace` that uses outputs of `torch.jit.trace` (does not always work). See [examples.ipynb](examples.ipynb).

## Acknowledgements

The script was moved from [functional-zoo](https://github.com/szagoruyko/functional-zoo) where it was created with the help of Adam Paszke, Soumith Chintala, Anton Osokin, and uses bits from [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch).
