---
title: Graph
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/graph.py#L245-L439 >}}

W&B class for graphs.

This class is typically used for saving and displaying neural net models.
It represents the graph as an array of nodes and edges. The nodes can have
labels that can be visualized by wandb.

#### Examples:

Import a keras model.

```python
import wandb

wandb.Graph.from_keras(keras_model)
```

## Methods

### `add_edge`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/graph.py#L354-L362)

```python
add_edge(
    from_node, to_node
)
```

Add an edge to the graph.

<!-- lazydoc-ignore: internal -->


### `add_node`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/graph.py#L338-L352)

```python
add_node(
    node=None, **node_kwargs
)
```

Add a node to the graph.

<!-- lazydoc-ignore: internal -->


### `from_keras`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/graph.py#L364-L400)

```python
@classmethod
from_keras(
    model
)
```

Create a graph from a Keras model.

This method is not supported for Keras 3.0.0 and above.
Requires a refactor.

"<!-- lazydoc-ignore-classmethod: internal -->

### `pprint`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/graph.py#L328-L336)

```python
pprint()
```

Pretty print the graph.

<!-- lazydoc-ignore: internal -->


### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/data_types/graph.py#L325-L326)

```python
__getitem__(
    nid
)
```
