---
title: run
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Runs。 

このモジュールは、W&B の Run および関連データとのやり取りのためのクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# フィルタに一致する run を取得
runs = Api().runs(
     path="entity/project", filters={"state": "finished", "config.batch_size": 32}
)

# run のデータへアクセス
for run in runs:
     print(f"Run: {run.name}")
     print(f"Config: {run.config}")
     print(f"Metrics: {run.summary}")

     # pandas で履歴データを取得
     history_df = run.history(keys=["loss", "accuracy"], pandas=True)

     # Artifact を操作
     for artifact in run.logged_artifacts():
         print(f"Artifact: {artifact.name}")
``` 



**注記:**

> このモジュールは W&B Public API の一部であり、run データへの読み書きアクセスを提供します。新しい run のログ記録には、メインの wandb パッケージの wandb.init() 関数を使用してください。

## <kbd>class</kbd> `Runs`
プロジェクトに関連する run と、必要ならフィルターで選ばれた run のイテラブルコレクションです。

通常は `Api.runs` 名前空間を介して間接的に利用します。



**引数:**
 
 - `client`:  (`wandb.apis.public.RetryingClient`) リクエストに使用する API クライアント。
 - `entity`:  (str) プロジェクトの所有者となる entity（ユーザー名またはチーム名）。
 - `project`:  (str) run を取得するプロジェクト名。
 - `filters`:  (Optional[Dict[str, Any]]) run クエリのために適用するフィルタの辞書。
 - `order`:  (Optional[str]) run の順序。"asc" または "desc"（デフォルトは "desc"）。
 - `per_page`:  (int) 1リクエストで取得する run の数（デフォルトは50）。
 - `include_sweeps`:  (bool) run に sweep の情報を含めるかどうか（デフォルトは True）。



**例:**
 ```python
from wandb.apis.public.runs import Runs
from wandb.apis.public import Api

# フィルタを満たすプロジェクト内のすべての run を取得
filters = {"state": "finished", "config.optimizer": "adam"}

runs = Api().runs(
    client=api.client,
    entity="entity",
    project="project_name",
    filters=filters,
)

# run を繰り返し処理し、詳細情報を表示
for run in runs:
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")
    print(f"Run state: {run.state}")
    print(f"Run config: {run.config}")
    print(f"Run summary: {run.summary}")
    print(f"Run history (samples=5): {run.history(samples=5)}")
    print("----------")

# 指定したメトリクスの履歴をすべての run から取得
histories_df = runs.histories(
    samples=100,  # 各 run から取得するサンプル数
    keys=["loss", "accuracy"],  # 取得するメトリクス
    x_axis="_step",  # x軸となるメトリクス
    format="pandas",  # pandas DataFrame 形式で返す
)
``` 

### <kbd>method</kbd> `Runs.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: Optional[str] = None,
    per_page: int = 50,
    include_sweeps: bool = True
)
```






---


### <kbd>property</kbd> Runs.length





---



### <kbd>method</kbd> `Runs.histories`

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = '_step',
    format: Literal['default', 'pandas', 'polars'] = 'default',
    stream: Literal['default', 'system'] = 'default'
)
```

フィルター条件に一致するすべての run について、サンプリングした履歴メトリクスを返します。



**引数:**
 
 - `samples`:  各 run から返すサンプル数
 - `keys`:  特定のキーに関するメトリクスのみ返す
 - `x_axis`:  このメトリクスを x軸 として使用（デフォルトは _step）
 - `format`:  データの返却形式。"default"、"pandas"、"polars" から選択
 - `stream`:  "default" で通常のメトリクス、"system" でシステムメトリクス

**返却値:**
 
 - `pandas.DataFrame`:  `format="pandas"` の場合、history メトリクスの `pandas.DataFrame` を返す
 - `polars.DataFrame`:  `format="polars"` の場合、history メトリクスの `polars.DataFrame` を返す
 - `list of dicts`:  `format="default"` の場合、`run_id` キーを含む history メトリクスの dict のリスト

---

## <kbd>class</kbd> `Run`
entity および project に紐付く単一の run です。



**引数:**
 
 - `client`:  W&B API クライアント
 - `entity`:  run に関連付けられた entity
 - `project`:  run に関連付けられた project
 - `run_id`:  run のユニーク識別子
 - `attrs`:  run の属性
 - `include_sweeps`:  sweep を run に含めるかどうか



**属性:**
 
 - `tags` ([str]):  run に紐づいたタグの一覧
 - `url` (str):  この run の URL
 - `id` (str):  run のユニークな識別子（デフォルトは8文字）
 - `name` (str):  run の名前
 - `state` (str):  状態（running, finished, crashed, killed, preempting, preempted のいずれか）
 - `config` (dict):  run に関連するハイパーパラメーターの辞書
 - `created_at` (str):  run 開始時刻（ISO タイムスタンプ）
 - `system_metrics` (dict):  run で記録された最新のシステムメトリクス
 - `summary` (dict):  現在の summary を保持するミュータブルな dict ライクプロパティ。update を呼ぶことで変更を保存できる
 - `project` (str):  run に関連付けられた project
 - `entity` (str):  run に関連付けられた entity の名前
 - `project_internal_id` (int):  プロジェクトの内部 ID
 - `user` (str):  run を作成したユーザー名
 - `path` (str):  一意の識別子 [entity]/[project]/[run_id]
 - `notes` (str):  run に関するノート
 - `read_only` (boolean):  run を編集できるかどうか
 - `history_keys` (str):  ログ済み履歴メトリクスのキー
 - `with `wandb.log({key`:  value})`
 - `metadata` (str):  wandb-metadata.json からの run のメタデータ

### <kbd>method</kbd> `Run.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: str,
    project: str,
    run_id: str,
    attrs: Optional[Mapping] = None,
    include_sweeps: bool = True
)
```

Run オブジェクトを初期化します。

Run は必ず wandb.Api のインスタンスである api の api.runs() を呼び出して初期化されます。

---

### <kbd>property</kbd> Run.entity

この run に関連付けられている entity。

---

### <kbd>property</kbd> Run.id

run の一意な識別子。

---

### <kbd>property</kbd> Run.lastHistoryStep

run の履歴内で最後に記録された step を返します。

---

### <kbd>property</kbd> Run.metadata

wandb-metadata.json からの run のメタデータ。

メタデータには run の説明、タグ、開始時刻、メモリ使用量などが含まれます。

---

### <kbd>property</kbd> Run.name

run の名前。

---

### <kbd>property</kbd> Run.path

run のパス。このパスは entity、project、run_id を含むリストです。

---

### <kbd>property</kbd> Run.state

run の状態。"Finished"、"Failed"、"Crashed"、"Running" のいずれかです。

---

### <kbd>property</kbd> Run.storage_id

run のストレージ固有の識別子。

---

### <kbd>property</kbd> Run.summary

run に紐づけられた summary 値を保持するミュータブルな dict ライクプロパティ。

---

### <kbd>property</kbd> Run.url

run の URL。

run の URL は entity、project、run_id から生成されます。SaaS ユーザーの場合、`https://wandb.ai/entity/project/run_id` の形式になります。

---

### <kbd>property</kbd> Run.username

この API は非推奨です。かわりに `entity` を利用してください。



---

### <kbd>classmethod</kbd> `Run.create`

```python
create(
    api,
    run_id=None,
    project=None,
    entity=None,
    state: Literal['running', 'pending'] = 'running'
)
```

指定したプロジェクトに run を作成します。

---

### <kbd>method</kbd> `Run.delete`

```python
delete(delete_artifacts=False)
```

wandb バックエンドから指定した run を削除します。



**引数:**
 
 - `delete_artifacts` (bool, optional):  run に関連付けられた Artifact も削除するか

---

### <kbd>method</kbd> `Run.file`

```python
file(name)
```

artifact 内で指定した名前のファイルのパスを返します。



**引数:**
 
 - `name` (str):  取得したいファイル名



**返却値:**
 `name` 引数に一致する `File` オブジェクト

---

### <kbd>method</kbd> `Run.files`

```python
files(names=None, per_page=50)
```

指定した名前ごとにファイルのパスを返します。



**引数:**
 
 - `names` (list):  取得したいファイルの名前、空の場合は全ファイルを返す
 - `per_page` (int):  1ページあたりの結果数



**返却値:**
 `Files` オブジェクト（`File` オブジェクトのイテレータ）

---

### <kbd>method</kbd> `Run.history`

```python
history(samples=500, keys=None, x_axis='_step', pandas=True, stream='default')
```

run の履歴メトリクスのサンプルを返します。

サンプルで良ければ、この方法が最もシンプルかつ高速です。



**引数:**
 
 - `samples `:  (int, optional) 返すサンプル数
 - `pandas `:  (bool, optional) pandas DataFrame で返すか
 - `keys `:  (list, optional) 特定のキーのメトリクスのみ返す
 - `x_axis `:  (str, optional) このメトリクスを x軸 とする（デフォルトは _step）
 - `stream `:  (str, optional) "default" で通常のメトリクス、"system" でシステムメトリクス



**返却値:**
 
 - `pandas.DataFrame`:  pandas=True の場合、history メトリクスの `pandas.DataFrame` を返す
 - `list of dicts`:  pandas=False の場合、history メトリクスの dict のリストを返す

---

### <kbd>method</kbd> `Run.load`

```python
load(force=False)
```





---

### <kbd>method</kbd> `Run.log_artifact`

```python
log_artifact(
    artifact: 'wandb.Artifact',
    aliases: Optional[Collection[str]] = None,
    tags: Optional[Collection[str]] = None
)
```

Artifact を run の出力として宣言します。



**引数:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)` から返された Artifact
 - `aliases` (list, optional):  この Artifact に付与するエイリアス
 - `tags`:  (list, optional) この Artifact に付与するタグ（任意）



**返却値:**
 `Artifact` オブジェクト

---

### <kbd>method</kbd> `Run.logged_artifacts`

```python
logged_artifacts(per_page: int = 100) → RunArtifacts
```

この run でログされたすべての Artifact を取得します。

run 中に記録されたすべての出力 Artifact を取得します。ページネーションされた結果が返り、イテレータやリストとして扱えます。



**引数:**
 
 - `per_page`:  1回の API リクエストで取得する Artifact の数



**返却値:**
 この run 中に出力 Artifact としてログされたすべての Artifact オブジェクトのイテラブルコレクション



**例:**
 ```python
import wandb
import tempfile

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
    tmp.write("This is a test artifact")
    tmp_path = tmp.name
run = wandb.init(project="artifact-example")
artifact = wandb.Artifact("test_artifact", type="dataset")
artifact.add_file(tmp_path)
run.log_artifact(artifact)
run.finish()

api = wandb.Api()

finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")

for logged_artifact in finished_run.logged_artifacts():
    print(logged_artifact.name)
``` 

---

### <kbd>method</kbd> `Run.save`

```python
save()
```

run オブジェクトへの変更を W&B バックエンドに保存します。

---

### <kbd>method</kbd> `Run.scan_history`

```python
scan_history(keys=None, page_size=1000, min_step=None, max_step=None)
```

run のすべての履歴レコードをイテラブルコレクションとして返します。



**引数:**
 
 - `keys` ([str], optional):  これらのキーだけ取得し、全てのキーが定義されている行のみ返す
 - `page_size` (int, optional):  API からまとめて取得するページサイズ
 - `min_step` (int, optional):  一度にスキャンする最小ステップ数
 - `max_step` (int, optional):  一度にスキャンする最大ステップ数



**返却値:**
 履歴レコード（dict）のイテラブルコレクション



**例:**
 特定の run からすべての loss 値をエクスポート

```python
run = api.run("entity/project-name/run-id")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
``` 

---

### <kbd>method</kbd> `Run.to_html`

```python
to_html(height=420, hidden=False)
```

この run を表示する iframe を生成する HTML を生成します。

---

### <kbd>method</kbd> `Run.update`

```python
update()
```

run オブジェクトへの変更を wandb バックエンドに保存します。

---

### <kbd>method</kbd> `Run.upload_file`

```python
upload_file(path, root='.')
```

ローカルファイルを W&B にアップロードし、この run に紐付けます。



**引数:**
 
 - `path` (str):  アップロードするファイルへのパス（絶対パスまたは相対パスが可）
 - `root` (str):  ファイル保存時の基準パス。例: run 内で "my_dir/file.txt" として保存したくて現在 "my_dir" にいる場合は、root を "../" に設定します。デフォルトは "."（カレントディレクトリ）



**返却値:**
 アップロードされたファイルを表す `File` オブジェクト

---

### <kbd>method</kbd> `Run.use_artifact`

```python
use_artifact(artifact, use_as=None)
```

Artifact を run の入力として宣言します。



**引数:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)` から返された Artifact
 - `use_as` (string, optional):  このスクリプト内で Artifact の用途を識別する文字列。beta 版の wandb launch 機能の Artifact 差し替え時の区別などに使用



**返却値:**
 `Artifact` オブジェクト

---

### <kbd>method</kbd> `Run.used_artifacts`

```python
used_artifacts(per_page: int = 100) → RunArtifacts
```

この run で明示的に使用された Artifact を取得します。

run 中に明示的に使用宣言された入力 Artifact（通常は `run.use_artifact()` で使用）が返されます。ページネーションされた結果が返り、イテレータやリストとして扱えます。



**引数:**
 
 - `per_page`:  1回の API リクエストで取得する Artifact の数



**返却値:**
 この run で入力として明示的に使用された Artifact オブジェクトのイテラブルコレクション



**例:**
 ```python
import wandb

run = wandb.init(project="artifact-example")
run.use_artifact("test_artifact:latest")
run.finish()

api = wandb.Api()
finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")
for used_artifact in finished_run.used_artifacts():
    print(used_artifact.name)
test_artifact
``` 

---

### <kbd>method</kbd> `Run.wait_until_finished`

```python
wait_until_finished()
```

run の状態が完了するまでチェックし続けます。