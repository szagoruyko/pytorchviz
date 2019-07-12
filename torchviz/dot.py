from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_dot(model, x):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        model: torch nn model
        x: input tensor
    """
    var = model(x)
    params = dict(model.named_parameters())
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var, grad=torch.zeros_like(var)):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                name = str(type(var).__name__).replace('Backward', '')
                node_name = '%s\n %s' % (name, size_to_str(grad.size()))
                if var in output_nodes:
                    dot.node(str(id(var)), node_name, fillcolor='darkolivegreen1')
                else:
                    dot.node(str(id(var)), node_name)

            seen.add(var)
            if hasattr(var, 'next_functions'):
                grads = var(grad)
                grads = grads if isinstance(grads, tuple) else [grads]
                for i, (u, grad) in enumerate(zip(var.next_functions, grads)):
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0], grad)
                    elif i == 0:
                        dot.edge(str(id(u[0])), str(id(var)))
                        node_name = 'input\n %s' % size_to_str(x.size())
                        dot.node(str(id(u[0])), node_name, fillcolor='orange')
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t, grad)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    resize_graph(dot)

    return dot


# For traces

def replace(name, scope):
    return '/'.join([scope[name], name])


def parse(graph):
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')
        inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': n.kind(),
                             'inputs': inputs,
                             'attr': attrs}))

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': 'Parameter',
                             'inputs': [],
                             'attr': str(n.type())}))

    return nodes


def make_dot_from_trace(trace):
    """ Produces graphs of torch.jit.trace outputs

    Example:
    >>> trace, = torch.jit.trace(model, args=(x,))
    >>> dot = make_dot_from_trace(trace)
    """
    # from tensorboardX
    if LooseVersion(torch.__version__) >= LooseVersion("0.4.1"):
        torch.onnx._optimize_trace(trace, torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    elif LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    graph = trace.graph()
    list_of_nodes = parse(graph)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in list_of_nodes:
        dot.node(node.name, label=node.name.replace('/', '\n'))
        if node.inputs:
            for inp in node.inputs:
                dot.edge(inp, node.name)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
