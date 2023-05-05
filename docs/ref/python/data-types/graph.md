# Graph



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/latest/wandb/data_types.py#L1375-L1536)



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



[View source](https://www.github.com/wandb/client/tree/latest/wandb/data_types.py#L1462-L1466)

```python
add_edge(
 from_node, to_node
)
```




### `add_node`



[View source](https://www.github.com/wandb/client/tree/latest/wandb/data_types.py#L1448-L1460)

```python
add_node(
 node=None, **node_kwargs
)
```




### `from_keras`



[View source](https://www.github.com/wandb/client/tree/latest/wandb/data_types.py#L1468-L1497)

```python
@classmethod
from_keras(
 model
)
```




### `pprint`



[View source](https://www.github.com/wandb/client/tree/latest/wandb/data_types.py#L1442-L1446)

```python
pprint()
```




### `__getitem__`



[View source](https://www.github.com/wandb/client/tree/latest/wandb/data_types.py#L1439-L1440)

```python
__getitem__(
 nid
)
```






