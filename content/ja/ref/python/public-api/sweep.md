---
title: Sweep
menu:
  reference:
    identifier: ja-ref-python-public-api-sweep
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L30-L240 >}}

スイープに関連付けられた一連の runs。

```python
Sweep(
    client, entity, project, sweep_id, attrs=None
)
```

#### 例:

次のようにインスタンス化します:

```
api = wandb.Api()
sweep = api.sweep(path / to / sweep)
```

| 属性 |  |
| :--- | :--- |
|  `runs` |  (`Runs`) run のリスト |
|  `id` |  (str) スイープの id |
|  `project` |  (str) プロジェクトの名前 |
|  `config` |  (str) スイープ設定の辞書 |
|  `state` |  (str) スイープの状態 |
|  `expected_run_count` |  (int) スイープの予想される run の数 |

## メソッド

### `best_run`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L125-L148)

```python
best_run(
    order=None
)
```

設定で定義されたメトリックまたは渡された順序でソートされた最良の run を返します。

### `display`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

このオブジェクトを jupyter に表示します。

### `get`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L173-L222)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

クラウドバックエンドに対してクエリを実行します。

### `load`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L106-L114)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L224-L232)

```python
to_html(
    height=420, hidden=(False)
)
```

このスイープを表示する iframe を含む HTML を生成します。

| クラス変数 |  |
| :--- | :--- |
|  `LEGACY_QUERY`<a id="LEGACY_QUERY"></a> |   |
|  `QUERY`<a id="QUERY"></a> |   |