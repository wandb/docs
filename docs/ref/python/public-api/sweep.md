# スイープ

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2487-L2697)

スイープに関連する一連のruns。

```python
Sweep(
 client, entity, project, sweep_id, attrs=None
)
```

#### 例:

次のようにインスタンス化します：
```
api = wandb.Api()
sweep = api.sweep(path/to/sweep)
```
| 属性 | |
| :--- | :--- |
| `runs` | (`Runs`) ランのリスト |
| `id` | (str) スイープID |
| `project` | (str) プロジェクト名 |
| `config` | (str) スイープ構成の辞書 |
| `state` | (str) スイープの状態 |
| `expected_run_count` | (int) スイープの予想ラン数 |



## メソッド

### `best_run`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2582-L2605)

```python
best_run(
 order=None
)
```

設定で定義されたメトリックまたは入力された順序で、最適なランを返します。
### `display`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L979-L990)

```python
display(
 height=420, hidden=(False)
) -> bool
```

このオブジェクトをjupyterで表示します。

### `get`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2630-L2679)

```python
@classmethod
get(
 client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

クラウドバックエンドに対してクエリを実行します。

### `load`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2563-L2571)

```python
load(
 force: bool = (False)
)
```

### `snake_to_camel`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L975-L977)

```python
snake_to_camel(
 string
)
```
### `to_html`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2681-L2689)

```python
to_html(
  height=420, hidden=(False)
)
```

このスイープを表示するiframeを含むHTMLを生成します。

| クラス変数 | |
| :--- | :--- |
| `LEGACY_QUERY` | |
| `QUERY` | |