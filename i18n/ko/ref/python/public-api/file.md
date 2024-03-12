
# 파일

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/files.py#L108-L194' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

파일은 wandb에 저장된 파일과 연결된 클래스입니다.

```python
File(
    client, attrs
)
```

| 속성 |  |
| :--- | :--- |

## 메소드

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/files.py#L174-L187)

```python
delete()
```

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 오브젝트를 jupyter에 표시합니다.

### `download`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/files.py#L134-L172)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False),
    api: Optional[Api] = None
) -> io.TextIOWrapper
```

wandb 서버에서 run에 의해 이전에 저장된 파일을 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  replace (boolean): `True`일 경우, 다운로드는 존재하는 로컬 파일을 덮어쓰게 됩니다. 기본값은 `False`입니다. root (str): 파일을 저장할 로컬 디렉토리. 기본값은 "."입니다. exist_ok (boolean): `True`일 경우, 파일이 이미 존재해도 ValueError를 발생시키지 않으며 replace=True가 아닌 이상 재다운로드하지 않습니다. 기본값은 `False`입니다. |

| 예외 |  |
| :--- | :--- |
|  파일이 이미 존재하고, replace=False 및 exist_ok=False인 경우 `ValueError`. |

### `snake_to_camel`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/attrs.py#L28-L29)

```python
to_html(
    *args, **kwargs
)
```