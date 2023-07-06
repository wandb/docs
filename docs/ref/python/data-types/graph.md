# Graph



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/v0.15.5/wandb/data_types.py#L1399-L1560)



Wandb class for graphs.

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



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/data_types.py#L1486-L1490)

```python
add_edge(
 from_node, to_node
)
```




### `add_node`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/data_types.py#L1472-L1484)

```python
add_node(
 node=None, **node_kwargs
)
```




### `from_keras`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/data_types.py#L1492-L1521)

```python
@classmethod
from_keras(
 model
)
```




### `pprint`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/data_types.py#L1466-L1470)

```python
pprint()
```




### `__getitem__`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/data_types.py#L1463-L1464)

```python
__getitem__(
 nid
)
```






