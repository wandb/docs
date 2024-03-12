
# 파일

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/files.py#L42-L105' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


`File` 오브젝트의 반복 가능한 컬렉션입니다.

```python
Files(
    client, run, names=None, per_page=50, upload=(False)
)
```

| 속성 |  |
| :--- | :--- |

## 메소드

### `convert_objects`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/files.py#L98-L102)

```python
convert_objects()
```

### `next`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/files.py#L95-L96)

```python
update_variables()
```

### `__getitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| 클래스 변수 |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |