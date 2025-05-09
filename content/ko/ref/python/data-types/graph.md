---
title: Graph
menu:
  reference:
    identifier: ko-ref-python-data-types-graph
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L245-L405 >}}

graphs를 위한 Wandb 클래스입니다.

```python
Graph(
    format="keras"
)
```

이 클래스는 일반적으로 신경망 모델을 저장하고 표시하는 데 사용됩니다. 노드와 엣지의 배열로 그래프를 나타냅니다. 노드는 wandb에서 시각화할 수 있는 레이블을 가질 수 있습니다.

#### 예시:

keras 모델 가져오기:

```
Graph.from_keras(keras_model)
```

## 메소드

### `add_edge`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L330-L334)

```python
add_edge(
    from_node, to_node
)
```

### `add_node`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L318-L328)

```python
add_node(
    node=None, **node_kwargs
)
```

### `from_keras`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L336-L366)

```python
@classmethod
from_keras(
    model
)
```

### `pprint`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L312-L316)

```python
pprint()
```

### `__getitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/graph.py#L309-L310)

```python
__getitem__(
    nid
)
```