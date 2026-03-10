# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import split_graph
from vllm.compilation.passes.fx_utils import find_op_nodes

# This import automatically registers `torch.ops.silly.attention`
from . import silly_attention  # noqa: F401


def test_simple():
    """
    Test simple model
    """
    def model_fn(x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        x = torch.relu(x)
        x = x + 2
        return x

    # Create a GraphModule using make_fx
    x = torch.randn(5, 3)
    gm = make_fx(model_fn)(x)

    # Split on relu, should create 3 submodules: [add], [relu], [add]
    split_ops = ["aten::relu"]
    split_gm, split_items = split_graph(gm, split_ops)
    assert len(split_items) == 3

    # Check correctness: outputs should match
    new_x = torch.randn(5, 3)
    output_original = gm(new_x)
    output_split = split_gm(new_x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_getitem_moved_to_producer_subgraph():
    """
    Test that getitem operations are moved to the same subgraph as their input,
    preventing tuple inputs to submodules.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        # torch.split returns a tuple, creating real getitem operations
        # Should become first submodule that produces tuple
        chunks = torch.split(x, x.shape[0] // 2, dim=0)

        # Following ops should become second submodule that consumes tuple
        result_0 = torch.relu(chunks[0])
        result_1 = torch.relu(chunks[1])
        return torch.cat([result_0, result_1], dim=0)

    # Create a GraphModule using make_fx
    x = torch.randn(4, 3)
    gm = make_fx(model_fn)(x)

    # Make sure the graph contains 'getitem' operations.
    has_getitem = any(
        node.op == "call_function" and node.target == operator.getitem
        for node in gm.graph.nodes
    )
    assert has_getitem, "Test setup failed: graph should contain getitem operations"

    # Split on tuple producer aten::split
    split_ops = ["aten::split.Tensor"]
    split_gm, split_items = split_graph(gm, split_ops)
    assert len(split_items) == 2, "Graph should be split into 2 submodules"

    # Check if submodule has any getitem directly applied to a placeholder 
    # (i.e., tuple input). There should be none.
    for split_item in split_items:
        submodule = split_item.graph
        getitem_on_placeholder = []
        for node in submodule.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.getitem
                and node.args[0].op == "placeholder"
            ):
                getitem_on_placeholder.append(node)
        assert len(getitem_on_placeholder) == 0, (
            f"Submodule {split_item.submod_name} has getitem operations on "
            f"placeholder nodes: {[n.name for n in getitem_on_placeholder]}. "
            "This means tuple inputs were not properly eliminated."
        )

    # Check correctness: outputs should match
    new_x = torch.randn(4, 3)
    output_original = gm(new_x)
    output_split = split_gm(new_x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_no_tuple_inputs_with_multiple_consumers():
    """
    Test that when a tuple is consumed by multiple split operations,
    getitem operations are properly moved to avoid tuple inputs.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        # torch.split returns a tuple, creating real getitem operations
        # Should become first submodule that produces tuple
        chunks = torch.split(x, x.shape[0] // 2, dim=0)

        # These should become second submodule consuming tuple
        result_1 = torch.relu(chunks[0])
        result_2 = torch.relu(chunks[1])

        # Artificial graph splitting point to create another
        # independent submodule that consumes tuple later
        # This would become the third submodule
        result_1 = torch.sigmoid(result_1)

        # Fourth submodule that consumes tuple
        result = torch.cat([chunks[0], chunks[1], result_1, result_2])
        return result

    # Create a GraphModule using make_fx
    x = torch.randn(4, 3)
    gm = make_fx(model_fn)(x)

    # Make sure the graph contains 'getitem' operations.
    has_getitem = any(
        node.op == "call_function" and node.target == operator.getitem
        for node in gm.graph.nodes
    )
    assert has_getitem, "Test setup failed: graph should contain getitem operations"

    split_ops = ["aten::split.Tensor", "aten::sigmoid"]
    split_gm, split_items = split_graph(gm, split_ops)
    assert len(split_items) == 4, "Graph should be split into 4 submodules"

    # Check if submodule has any getitem directly applied to a placeholder 
    # (i.e., tuple input). There should be none.
    for split_item in split_items:
        submodule = split_item.graph
        for node in submodule.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.getitem
                and node.args[0].op == "placeholder"
            ):
                pytest.fail(
                    f"Submodule {split_item.submod_name} has getitem on "
                    f"placeholder {node.args[0].name}, indicating it receives "
                    "a tuple input"
                )

    # Check correctness: outputs should match
    new_x = torch.randn(4, 3)
    output_original = gm(new_x)
    output_split = split_gm(new_x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_consecutive_ops_in_split():
    """
    Test that consecutive splitting operations are grouped into the same subgraph
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        intermediate = torch.relu(x)
        attn_inout = torch.sqrt(intermediate)
        torch.ops.silly.attention(intermediate, intermediate, attn_inout, attn_inout)
        final_result = torch.sigmoid(attn_inout)
        return final_result

    # Create a GraphModule using make_fx
    x = torch.randn(8, 4)
    gm = make_fx(model_fn)(x)

    # Check that relu and sqrt are present in the graph
    assert (
        len(list(find_op_nodes(torch.ops.aten.relu, gm.graph))) == 1
        and len(list(find_op_nodes(torch.ops.aten.sqrt, gm.graph))) == 1
    ), "Test setup failed: Expected sqrt and relu operations in the graph."

    # Split on silly::attention and aten::sqrt (should be grouped)
    splitting_ops = ["silly::attention", "aten::sqrt"]
    split_gm, split_items = split_graph(gm, splitting_ops)
    assert len(split_items) == 3, (
        "Consecutive splitting operations were not grouped correctly."
    )

    # Check correctness: outputs should match
    new_x = torch.randn(8, 4, device="cuda")
    output_original = gm(new_x)
    output_split = split_gm(new_x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"

    # Only one splitting graph (grouped consecutive ops)
    splitting_items = list(s for s in split_items if s.is_splitting_graph)
    assert len(splitting_items) == 1, "Expecting a single splitting graph"
    splitting_gm = splitting_items[0].graph
    
    # Should have 4 nodes: placeholder, sqrt, silly attention, output
    assert len(splitting_gm.graph.nodes) == 4, "Expecting 4 nodes in splitting graph"
    assert [node.op for node in splitting_gm.graph.nodes] == ["placeholder"] + 2 * [
        "call_function"
    ] + ["output"]


@pytest.mark.parametrize("tracer_type", ["symbolic", "make_fx"])
def test_empty_partition_is_merged(tracer_type: str):
    """
    Test that an empty-allocation-only partition is merged into its previous
    partition.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        y = torch.sin(x)
        out = torch.empty_like(y)
        torch.ops.aten.cos.out(y, out=out)
        return out

    # Create a GraphModule
    if tracer_type == "symbolic":
        # Test for builtin can be merged.
        gm = torch.fx.symbolic_trace(model_fn)
    else:
        # Test for aten::empty_like can be merged.
        x = torch.randn(4, 3)
        gm = make_fx(model_fn)(x)

    # Split on sin and cos
    split_ops = ["aten::sin", "aten::cos.out"]
    split_gm, split_items = split_graph(gm, split_ops)

    # Without the merge, this graph is split into 3 partitions: [sin], [empty_like], [cos].
    # With the merge, the empty_like is merged into the first partition, resulting in 2 partitions: [sin + empty_like], [cos].
    assert len(split_items) == 2, "Empty-only partition should be merged"

    # Check correctness: outputs should match
    x = torch.randn(4, 3)
    output_original = gm(x)
    output_split = split_gm(x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


@pytest.mark.parametrize("tracer_type", ["symbolic", "make_fx"])
def test_empty_only_partition_between_split_ops_is_merged(tracer_type: str):
    """
    Test that empty-only partitions between split ops are merged, but an initial empty-only partition is not merged.
    """
    def model_fn(x: torch.Tensor) -> torch.Tensor:
        out1 = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, out1)
        out2 = torch.empty_like(x)
        torch.ops.silly.attention(out1, out1, out1, out2)
        return out2

    # Create a GraphModule
    if tracer_type == "symbolic":
        # Test for builtin empty can be merged.
        gm = torch.fx.symbolic_trace(model_fn)
    else:
        # Test for aten::empty_like can be merged.
        x = torch.randn(4, 3)
        gm = make_fx(model_fn)(x)

    split_ops = ["silly::attention"]
    split_gm, split_items = split_graph(gm, split_ops)

    # Without the merge, this graph is split into 4 partitions: [empty_like], [attention], [empty_like], [attention], [empty_like]. 
    # With the merge, resulting in 3 partitions: [empty_like], [attention + empty_like], [attention].
    assert len(split_items) == 3, "Empty-only partition should be merged"

    # Check correctness: outputs should match
    x = torch.randn(2, 3, device="cuda")
    output_original = gm(x)
    output_split = split_gm(x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"
