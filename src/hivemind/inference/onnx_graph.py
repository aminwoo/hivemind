"""Graph ordering shared by ONNX export and precision conversion."""

import onnx


def topologically_sort_graph(model: onnx.ModelProto) -> None:
    available_values = {
        value.name for value in [*model.graph.input, *model.graph.initializer]
    }
    pending_nodes = list(model.graph.node)
    sorted_nodes = []
    while pending_nodes:
        ready_nodes = [
            node
            for node in pending_nodes
            if all(not name or name in available_values for name in node.input)
        ]
        if not ready_nodes:
            raise RuntimeError("FP16 ONNX conversion produced an unsortable graph")
        for node in ready_nodes:
            sorted_nodes.append(node)
            available_values.update(name for name in node.output if name)
            pending_nodes.remove(node)

    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)


