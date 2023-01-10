# Graph



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1380-L1541)



Wandb class for graphs

```python
Graph(
 format="keras"
)
```




This class is typically used for saving and diplaying neural net models. It
represents the graph as an array of nodes and edges. The nodes can have
labels that can be visualized by wandb.

#### Examples:

Import a keras model:
```
 Graph.from_keras(keras_model)
```



## Methods

### `add_edge`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1467-L1471)

```python
add_edge(
 from_node, to_node
)
```




### `add_node`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1453-L1465)

```python
add_node(
 node=None, \*\*node_kwargs
)
```




### `from_keras`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1473-L1502)

```python
@classmethod
from_keras(
 model
)
```




### `pprint`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1447-L1451)

```python
pprint()
```




### `__getitem__`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1444-L1445)

```python
__getitem__(
 nid
)
```






