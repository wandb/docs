# グラフ

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1389-L1550)

Wandbのグラフ用クラスです。

```python
Graph(
 format="keras"
)
```

このクラスは通常、ニューラルネットワークのモデルを保存および表示するために使用されます。グラフはノードとエッジの配列として表現されます。ノードには、wandbによって可視化できるラベルを付けることができます。

#### 例：

Kerasモデルをインポートします：
```
 Graph.from_keras(keras_model)
```
## メソッド

### `add_edge`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1476-L1480)

```python
add_edge(
 from_node, to_node
)
```

### `add_node`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1462-L1474)

```python
add_node(
 node=None, **node_kwargs
)
```
### `from_keras`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1482-L1511)

```python
@classmethod
from_keras(
 モデル
)
```




### `pprint`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1456-L1460)

```python
pprint()
```
### `__getitem__`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1453-L1454)

```python
__getitem__(
  nid
)
```