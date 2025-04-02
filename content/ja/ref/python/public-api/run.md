---
title: Run
menu:
  reference:
    identifier: ja-ref-python-public-api-run
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L276-L1007 >}}

ある entity と project に関連付けられた単一の run。

```python
Run(
    client: "RetryingClient",
    entity: str,
    project: str,
    run_id: str,
    attrs: Optional[Mapping] = None,
    include_sweeps: bool = (True)
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `create`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L377-L417)

```python
@classmethod
create(
    api, run_id=None, project=None, entity=None
)
```

指定された project の run を作成します。

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L540-L568)

```python
delete(
    delete_artifacts=(False)
)
```

指定された run を wandb バックエンドから削除します。

### `display`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

このオブジェクトを jupyter で表示します。

### `file`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L632-L642)

```python
file(
    name
)
```

アーティファクト内の指定された名前のファイルのパスを返します。

| Args |  |
| :--- | :--- |
|  name (str): リクエストされたファイルの名前。 |

| Returns |  |
| :--- | :--- |
|  name 引数に一致する `File`。 |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L619-L630)

```python
files(
    names=None, per_page=50
)
```

名前が指定された各ファイルのファイルパスを返します。

| Args |  |
| :--- | :--- |
|  names (list): リクエストされたファイルの名前。空の場合、すべてのファイルを返します。 per_page (int): ページごとの結果数。 |

| Returns |  |
| :--- | :--- |
|  `File` オブジェクトのイテレーターである `Files` オブジェクト。 |

### `history`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L668-L708)

```python
history(
    samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

run のサンプルされた履歴メトリクスを返します。

履歴レコードのサンプリングが問題ない場合は、これを使用するとより簡単かつ高速になります。

| Args |  |
| :--- | :--- |
|  `samples` |  (int, optional) 返すサンプル数 |
|  `pandas` |  (bool, optional) pandas DataFrame を返します |
|  `keys` |  (list, optional) 特定のキーのメトリクスのみを返します |
|  `x_axis` |  (str, optional) このメトリクスを xAxis として使用します。デフォルトは _step です |
|  `stream` |  (str, optional) メトリクスには "default"、マシンメトリクスには "system" |

| Returns |  |
| :--- | :--- |
|  `pandas.DataFrame` |  pandas=True の場合、履歴メトリクスの `pandas.DataFrame` を返します。 dict のリスト: pandas=False の場合、履歴メトリクスの dict のリストを返します。 |

### `load`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L419-L488)

```python
load(
    force=(False)
)
```

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L860-L905)

```python
log_artifact(
    artifact: "wandb.Artifact",
    aliases: Optional[Collection[str]] = None,
    tags: Optional[Collection[str]] = None
)
```

アーティファクトを run の出力として宣言します。

| Args |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)` から返されたアーティファクト。 aliases (list, optional): このアーティファクトに適用するエイリアス。 |
|  `tags` |  (list, optional) このアーティファクトに適用するタグ (存在する場合)。 |

| Returns |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `logged_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L766-L798)

```python
logged_artifacts(
    per_page: int = 100
) -> public.RunArtifacts
```

この run によって記録されたすべてのアーティファクトをフェッチします。

run 中に記録されたすべての出力 Artifacts を取得します。反復処理したり、単一のリストに収集したりできるページ分割された結果を返します。

| Args |  |
| :--- | :--- |
|  `per_page` |  API リクエストごとにフェッチする Artifacts の数。 |

| Returns |  |
| :--- | :--- |
|  この run 中に出力として記録されたすべての Artifact オブジェクトの反復可能なコレクション。 |

#### Example:

```
>>> import wandb
>>> import tempfile
>>> with tempfile.NamedTemporaryFile(
...     mode="w", delete=False, suffix=".txt"
... ) as tmp:
...     tmp.write("This is a test artifact")
...     tmp_path = tmp.name
>>> run = wandb.init(project="artifact-example")
>>> artifact = wandb.Artifact("test_artifact", type="dataset")
>>> artifact.add_file(tmp_path)
>>> run.log_artifact(artifact)
>>> run.finish()
>>> api = wandb.Api()
>>> finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")
>>> for logged_artifact in finished_run.logged_artifacts():
...     print(logged_artifact.name)
test_artifact
```

### `save`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L570-L571)

```python
save()
```

### `scan_history`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L710-L764)

```python
scan_history(
    keys=None, page_size=1000, min_step=None, max_step=None
)
```

run のすべての履歴レコードの反復可能なコレクションを返します。

#### Example:

サンプル run のすべての損失値をエクスポートします。

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

| Args |  |
| :--- | :--- |
|  keys ([str], optional): これらのキーのみをフェッチし、すべてのキーが定義されている行のみをフェッチします。 page_size (int, optional): API からフェッチするページのサイズ。 min_step (int, optional): 一度にスキャンする最小ページ数。 max_step (int, optional): 一度にスキャンする最大ページ数。 |

| Returns |  |
| :--- | :--- |
|  履歴レコード (dict) の反復可能なコレクション。 |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L993-L1001)

```python
to_html(
    height=420, hidden=(False)
)
```

この run を表示する iframe を含む HTML を生成します。

### `update`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L512-L538)

```python
update()
```

run オブジェクトへの変更を wandb バックエンドに永続化します。

### `upload_file`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L644-L666)

```python
upload_file(
    path, root="."
)
```

ファイルをアップロードします。

| Args |  |
| :--- | :--- |
|  path (str): アップロードするファイルの名前。 root (str): ファイルを相対的に保存するルートパス。 例: ファイルを run に "my_dir/file.txt" として保存し、現在 "my_dir" にいる場合は、root を "../" に設定します。 |

| Returns |  |
| :--- | :--- |
|  name 引数に一致する `File`。 |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L827-L858)

```python
use_artifact(
    artifact, use_as=None
)
```

アーティファクトを run への入力として宣言します。

| Args |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)` から返されたアーティファクト use_as (string, optional): スクリプトでアーティファクトがどのように使用されるかを識別する文字列。 ベータ版の wandb launch 機能のアーティファクトスワップ機能を使用する場合に、run で使用されるアーティファクトを簡単に区別するために使用されます。 |

| Returns |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `used_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L800-L825)

```python
used_artifacts(
    per_page: int = 100
) -> public.RunArtifacts
```

この run で明示的に使用されている Artifacts をフェッチします。

通常は `run.use_artifact()` を介して、run 中に使用されたと明示的に宣言された入力 Artifacts のみを取得します。反復処理したり、単一のリストに収集したりできるページ分割された結果を返します。

| Args |  |
| :--- | :--- |
|  `per_page` |  API リクエストごとにフェッチする Artifacts の数。 |

| Returns |  |
| :--- | :--- |
|  この run で入力として明示的に使用される Artifact オブジェクトの反復可能なコレクション。 |

#### Example:

```
>>> import wandb
>>> run = wandb.init(project="artifact-example")
>>> run.use_artifact("test_artifact:latest")
>>> run.finish()
>>> api = wandb.Api()
>>> finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")
>>> for used_artifact in finished_run.used_artifacts():
...     print(used_artifact.name)
test_artifact
```

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L490-L510)

```python
wait_until_finished()
```