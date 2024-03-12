
# 그래프

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1326-L1485' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


그래프에 대한 Wandb 클래스입니다.

```python
Graph(
    format="keras"
)
```

이 클래스는 일반적으로 신경망 모델을 저장하고 표시하는 데 사용됩니다. 노드와 엣지의 배열로 그래프를 나타냅니다. 노드는 wandb에 의해 시각화될 수 있는 레이블을 가질 수 있습니다.

#### 예시:

Keras 모델 가져오기:

```
Graph.from_keras(keras_model)
```

## 메소드

### `add_edge`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1411-L1415)

```python
add_edge(
    from_node, to_node
)
```

### `add_node`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1399-L1409)

```python
add_node(
    node=None, **node_kwargs
)
```

### `from_keras`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1417-L1446)

```python
@classmethod
from_keras(
    model
)
```

### `pprint`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1393-L1397)

```python
pprint()
```

### `__getitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1390-L1391)

```python
__getitem__(
    nid
)
```