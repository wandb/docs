---
title: ファイル
menu:
  reference:
    identifier: ja-ref-python-public-api-files
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/files.py#L44-L107 >}}

`File` オブジェクトの反復可能なコレクション。

```python
Files(
    client, run, names=None, per_page=50, upload=(False)
)
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `convert_objects`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/files.py#L100-L104)

```python
convert_objects()
```

### `next`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/files.py#L97-L98)

```python
update_variables()
```

### `__getitem__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| クラス変数 |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |