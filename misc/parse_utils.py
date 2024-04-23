from dataclasses import dataclass, field
from typing import NewType, Optional

### Class Definition

# Deletion Algorithm needs an interface like this class:
#
# class Node:
#    def __init__(self, children=None):
#        if children is None:
#            children = []
#        self.children = children


# For compatibility with newick parser:

NodeName = NewType("NodeName", str)


@dataclass
class Node:
    name: NodeName = field(default=NodeName(""))
    length: Optional[float] = None
    children: list["Node"] = field(default_factory=list)
    is_null: bool = False


def new_node(stack, buffer, mode):
    new_node = Node()
    stack[-1].children.append(new_node)
    stack.append(new_node)
    return stack, buffer, mode


def close_node(stack, buffer, mode):
    stack.pop()
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


### Newick Parser
def flush_buffer(stack, buffer, mode, to_numeric=False):
    if mode == "name" and buffer:
        stack[-1].name = "".join(buffer)
        buffer = []
    elif mode == "length" and buffer:
        stack[-1].length = float("".join(buffer))
        buffer = []
    mode = "length" if to_numeric else "name"
    return stack, buffer, mode
