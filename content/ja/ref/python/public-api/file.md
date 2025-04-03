---
title: File
menu:
  reference:
    identifier: ja-ref-python-public-api-file
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/files.py#L110-L263 >}}

File は wandb によって保存されたファイルに関連付けられたクラスです。

```python
File(
    client, attrs, run=None
)
```

| Attributes |  |
| :--- | :--- |
|  `path_uri` |  ストレージ バケット内のファイルへの URI パスを返します。 |

## メソッド

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

このオブジェクトを jupyter で表示します。

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

以前に run によって保存されたファイルを wandb サーバー からダウンロードします。

| Args |  |
| :--- | :--- |
|  replace (boolean): `True` の場合、ダウンロードはローカル ファイルが存在する場合に上書きします。デフォルトは `False` です。 root (str): ファイルを保存するローカル ディレクトリー。デフォルトは "." です。 exist_ok (boolean): `True` の場合、ファイルが既に存在する場合は ValueError を発生させず、replace=True でない限り再ダウンロードしません。デフォルトは `False` です。 api (Api, optional): 指定された場合、ファイルのダウンロードに使用される `Api` インスタンス。 |

| Raises |  |
| :--- | :--- |
|  ファイルが既に存在し、replace=False かつ exist_ok=False の場合、`ValueError`。 |

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