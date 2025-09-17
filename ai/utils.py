import asyncio
import numpy as np
from ai.graph import G


# draw the Computational Graph of the ai program
def draw_graph(filename=None, format='svg', graph=G):
    # visualization procedure referred from karpathy's micrograd

    from graphviz import Digraph

    label = 'Computational Graph of {}'.format(filename)
    dot = Digraph(graph_attr={'rankdir': 'LR', 'label': label}, node_attr={'rankdir': 'TB'})

    for cell in graph.nodes:

        # add the op to nodes
        dot.node(name=str(id(cell.backward_op)), label=cell.op, shape='doublecircle',)

        for input in cell.inputs:

            # add the input to nodes
            color = None if input.requires_grad else 'red'
            dot.node(name=str(id(input)), label='{}'.format(input.node_id), shape='circle', color=color)
            # forward pass edge from input to op
            dot.edge(str(id(input)), str(id(cell.backward_op)))

            # # backprop pass edge from op to input
            # if input.requires_grad:
            #     dot.edge(str(id(cell.backward_op)), str(id(input)), color='red')

        for output in cell.outputs:

            # add the output to nodes
            dot.node(name=str(id(output)), label='{}'.format(output.node_id), shape='circle')
            # forward pass edge from op to output
            dot.edge(str(id(cell.backward_op)), str(id(output)))

            # # backward pass edge from output to op
            # dot.edge(str(id(output)), str(id(cell.backward_op)), color='red')

    dot.render(format=format, filename=filename, directory='assets', cleanup=True)


# clip the gradients of parameters by value
def clip_grad_value(parameters, clip_value):

    for p in parameters:
        # clip gradients by value
        p.grad = np.clip(p.grad, -clip_value, clip_value)


# async utils from hummingbot
# ref:  https://github.com/hummingbot/hummingbot/blob/master/hummingbot/core/utils/async_utils.py
async def safe_wrapper(c):
    try:
        return await c
    except asyncio.CancelledError:
        raise
    except Exception as e:
        raise Exception(f"Unhandled error in background task: {str(e)}")


def safe_ensure_future(coro, *args, **kwargs):
    return asyncio.ensure_future(safe_wrapper(coro), *args, **kwargs)


async def safe_gather(*args, **kwargs):
    try:
        return await asyncio.gather(*args, **kwargs)
    except Exception as e:
        raise Exception(f"Unhandled error in background task: {str(e)}")