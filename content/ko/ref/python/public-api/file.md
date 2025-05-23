---
title: File
menu:
  reference:
    identifier: ko-ref-python-public-api-file
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/files.py#L110-L263 >}}

File은 wandb에 의해 저장된 파일과 연결된 클래스입니다.

```python
File(
    client, attrs, run=None
)
```

| 속성 |  |
| :--- | :--- |
|  `path_uri` | 스토리지 버킷에 있는 파일의 uri 경로를 반환합니다. |

## 메소드

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/files.py#L193-L223)

```python
delete()
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Jupyter에서 이 오브젝트를 표시합니다.

### `download`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/files.py#L152-L191)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False),
    api: Optional[Api] = None
) -> io.TextIOWrapper
```

wandb 서버에서 run에 의해 이전에 저장된 파일을 다운로드합니다.

| Args |  |
| :--- | :--- |
|  replace (boolean): `True`인 경우, 다운로드는 로컬 파일이 존재하면 덮어씁니다. 기본값은 `False`입니다. root (str): 파일을 저장할 로컬 디렉토리입니다. 기본값은 "."입니다. exist_ok (boolean): `True`인 경우, 파일이 이미 존재하면 ValueError를 발생시키지 않으며 replace=True가 아니면 다시 다운로드하지 않습니다. 기본값은 `False`입니다. api (Api, optional): 주어진 경우, 파일을 다운로드하는 데 사용되는 `Api` 인스턴스입니다. |

| Raises |  |
| :--- | :--- |
|  파일이 이미 존재하고 replace=False 및 exist_ok=False인 경우 `ValueError`가 발생합니다. |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L39-L40)

```python
to_html(
    *args, **kwargs
)
```