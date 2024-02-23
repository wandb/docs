
# 파일

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/files.py#L106-L185' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

파일은 wandb에 의해 저장된 파일과 관련된 클래스입니다.

```python
File(
    client, attrs
)
```

| 속성 |  |
| :--- | :--- |

## 메서드

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/files.py#L165-L178)

```python
delete()
```

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 개체를 jupyter에서 표시합니다.

### `download`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/files.py#L132-L163)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False)
) -> io.TextIOWrapper
```

wandb 서버에서 실행에 의해 이전에 저장된 파일을 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  replace (boolean): `True`일 경우, 로컬 파일이 존재하는 경우 다운로드가 해당 파일을 덮어씁니다. 기본값은 `False`입니다. root (str): 파일을 저장할 로컬 디렉터리. 기본값은 "."입니다. exist_ok (boolean): `True`일 경우, 파일이 이미 존재해도 ValueError를 발생시키지 않으며 replace=True가 아닌 경우 재다운로드하지 않습니다. 기본값은 `False`입니다. |

| 예외 |  |
| :--- | :--- |
|  파일이 이미 존재하고 replace=False, exist_ok=False 일 경우 `ValueError` 발생. |

### `snake_to_camel`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/attrs.py#L28-L29)

```python
to_html(
    *args, **kwargs
)
```