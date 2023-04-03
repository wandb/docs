# Graph



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/data_types.py#L1386-L1547)



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



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/data_types.py#L1473-L1477)

```python
add_edge(
 from_node, to_node
)
```




### `add_node`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/data_types.py#L1459-L1471)

```python
add_node(
 node=None, **node_kwargs
)
```




### `from_keras`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/data_types.py#L1479-L1508)

```python
@classmethod
from_keras(
 model
)
```




### `pprint`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/data_types.py#L1453-L1457)

```python
pprint()
```




### `__getitem__`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/data_types.py#L1450-L1451)

```python
__getitem__(
 nid
)
```






