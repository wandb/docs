# File

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/files.py#L108-L195' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

File은 wandb에 의해 저장된 파일과 관련된 클래스입니다.

```python
File(
    client, attrs
)
```

| 속성 |  |
| :--- | :--- |

## 메소드

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/files.py#L175-L188)

```python
delete()
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 오브젝트를 jupyter에서 표시합니다.

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/files.py#L134-L173)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False),
    api: Optional[Api] = None
) -> io.TextIOWrapper
```

wandb 서버의 run에 의해 이전에 저장된 파일을 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  replace (boolean): `True`이면, 다운로드는 로컬 파일이 존재할 경우 해당 파일을 덮어씁니다. 기본값은 `False`입니다. root (str): 파일을 저장할 로컬 디렉토리입니다. 기본값은 "."입니다. exist_ok (boolean): `True`이면, 파일이 이미 존재해도 ValueError를 발생시키지 않으며, replace=True가 아닌 한 재다운로드하지 않습니다. 기본값은 `False`입니다. api (Api, optional): 제공된 경우 파일을 다운로드하는 데 사용되는 `Api` 인스턴스입니다. |

| 예외 |  |
| :--- | :--- |
|  파일이 이미 존재하고, replace=False이며 exist_ok=False인 경우 `ValueError`를 발생시킵니다. |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/attrs.py#L28-L29)

```python
to_html(
    *args, **kwargs
)
```