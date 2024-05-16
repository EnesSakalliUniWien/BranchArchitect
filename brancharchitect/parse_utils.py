from dataclasses import dataclass, field
from typing import NewType, Optional

from brancharchitect.node import Node

__all__ = ['parse_newick']


def new_node(stack, buffer, mode):
    new_node = Node()
    stack[-1].children.append(new_node)
    stack.append(new_node)
    return stack, buffer, mode


def close_node(stack, buffer, mode):
    stack.pop()
    return stack, buffer, mode


def flush_buffer(stack, buffer, mode, to_numeric=False):
    if mode == "name" and buffer:
        stack[-1].name = "".join(buffer)
        buffer = []
    elif mode == "length" and buffer:
        stack[-1].length = float("".join(buffer))
        buffer = []
    mode = "length" if to_numeric else "name"
    return stack, buffer, mode


def parse_newick(newick_string: str):
    stack = [Node()]
    buffer: list[str] = []
    mode = "name"

    for char in newick_string:
        if char == "(":
            stack, buffer, mode = flush_buffer(stack, buffer, mode)
            stack, buffer, mode = new_node(stack, buffer, mode)
        elif char == ")":
            stack, buffer, mode = flush_buffer(stack, buffer, mode)
            stack, buffer, mode = close_node(stack, buffer, mode)
        elif char == ",":
            stack, buffer, mode = flush_buffer(stack, buffer, mode)
            stack, buffer, mode = close_node(stack, buffer, mode)
            stack, buffer, mode = new_node(stack, buffer, mode)
        elif char == ":":
            stack, buffer, mode = flush_buffer(stack, buffer, mode, to_numeric=True)
        elif char == ";":
            stack, buffer, mode = flush_buffer(stack, buffer, mode)
            break
        else:
            buffer.append(char)
    assert len(stack) == 1
    return stack[0]
