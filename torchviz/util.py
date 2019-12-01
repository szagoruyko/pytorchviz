import torch

from . import dot

def trace_and_dot(model, x):
    """Runs a trace using onnx to generate a graph.
    
    Args:
        model: The model to trace
        x: The input variable to trace.
    """
    with torch.onnx.set_training(model, False):
        trace, _ = torch.jit.get_trace_graph(model, args=(x,))
    return dot.make_dot_from_trace(trace)
