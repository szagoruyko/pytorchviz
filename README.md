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
pip install git+https://github.com/szagoruyko/pytorchviz
```


## Usage

<img width="891" alt="screen shot 2018-01-30 at 16 13 01" src="https://user-images.githubusercontent.com/4953728/35574234-8780297e-05d9-11e8-8e80-f4009297cefd.png">

There are two functions, `make_dot` to make graphs from any PyTorch functions (requires that at least one input Variable requires_grad), and `make_dot_from_trace` that uses outputs of `torch.jit.trace` (does not always work). See [examples.ipynb](examples.ipynb).

## Acknowledgements

The script was moved from [functional-zoo](https://github.com/szagoruyko/functional-zoo) where it was created with the help of Adam Paszke, Soumith Chintala, Anton Osokin, and uses bits from [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch).
