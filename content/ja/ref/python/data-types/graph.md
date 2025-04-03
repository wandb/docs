---
title: Graph
menu:
  reference:
    identifier: ja-ref-python-data-types-graph
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L245-L405 >}}

グラフのための Wandb クラス。

```python
Graph(
    format="keras"
)
```

このクラスは通常、ニューラルネットのモデルを保存および表示するために使用されます。これは、ノードとエッジの配列としてグラフを表します。ノードは wandb で可視化できるラベルを持つことができます。

#### 例：

keras モデルをインポートします：

```
Graph.from_keras(keras_model)
```

## メソッド

### `add_edge`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L330-L334)

```python
add_edge(
    from_node, to_node
)
```

### `add_node`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L318-L328)

```python
add_node(
    node=None, **node_kwargs
)
```

### `from_keras`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L336-L366)

```python
@classmethod
from_keras(
    model
)
```

### `pprint`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L312-L316)

```python
pprint()
```

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L309-L310)

```python
__getitem__(
    nid
)
```