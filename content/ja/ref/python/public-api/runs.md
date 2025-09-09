---
title: runs
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-runs
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B の Runs 向け Public API。 

このモジュールは、W&B の run とその関連 データ に対して操作するためのクラスを提供します。 



**例:**
 ```python
from wandb.apis.public import Api

# フィルターに一致する run を取得
runs = Api().runs(
     path="entity/project", filters={"state": "finished", "config.batch_size": 32}
)

# run データへアクセス
for run in runs:
     print(f"Run: {run.name}")
     print(f"Config: {run.config}")
     print(f"Metrics: {run.summary}")

     # pandas で履歴を取得
     history_df = run.history(keys=["loss", "accuracy"], pandas=True)

     # Artifacts を扱う
     for artifact in run.logged_artifacts():
         print(f"Artifact: {artifact.name}")
``` 



**注記:**

> このモジュールは W&B Public API の一部で、run データへの読み書き アクセス を提供します。新しい run を ログ する場合は、メインの wandb パッケージにある wandb.init() 関数を使用してください。 

## <kbd>class</kbd> `Runs`
project と任意のフィルターに関連付けられた `Run` オブジェクトの遅延イテレーター。 

必要に応じて W&B サーバー からページ単位で Runs が取得されます。 

通常は `Api.runs` 名前空間を通じて間接的に使用します。 



**引数:**
 
 - `client`:  (`wandb.apis.public.RetryingClient`) リクエストに使用する API クライアント。 
 - `entity`:  (str) project を所有する entity（ユーザー名または team）。 
 - `project`:  (str) run を取得する対象の project 名。 
 - `filters`:  (Optional[Dict[str, Any]]) run クエリに適用するフィルターの 辞書。 
 - `order`:  (str) 並び順。`created_at`、`heartbeat_at`、`config.*.value`、`summary_metrics.*` を指定できます。先頭に + を付けると昇順（デフォルト）、- を付けると降順。デフォルトは古い順の run.created_at。 
 - `per_page`:  (int) 1 リクエストで取得する run 数（デフォルト 50）。 
 - `include_sweeps`:  (bool) Runs に Sweep 情報を含めるかどうか。デフォルトは True。 



**例:**
 ```python
from wandb.apis.public.runs import Runs
from wandb.apis.public import Api

# フィルターを満たす project 内のすべての run を取得
filters = {"state": "finished", "config.optimizer": "adam"}

runs = Api().runs(
    client=api.client,
    entity="entity",
    project="project_name",
    filters=filters,
)

# run を反復し詳細を表示
for run in runs:
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")
    print(f"Run state: {run.state}")
    print(f"Run config: {run.config}")
    print(f"Run summary: {run.summary}")
    print(f"Run history (samples=5): {run.history(samples=5)}")
    print("----------")

# 特定のメトリクスについて全 run の履歴を取得
histories_df = runs.histories(
    samples=100,  # 各 run のサンプル数
    keys=["loss", "accuracy"],  # 取得するメトリクス
    x_axis="_step",  # X 軸に使うメトリクス
    format="pandas",  # pandas DataFrame で返す
)
``` 

### <kbd>method</kbd> `Runs.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: 'str',
    project: 'str',
    filters: 'dict[str, Any] | None' = None,
    order: 'str' = '+created_at',
    per_page: 'int' = 50,
    include_sweeps: 'bool' = True
)
```






---


### <kbd>property</kbd> Runs.length





---



### <kbd>method</kbd> `Runs.histories`

```python
histories(
    samples: 'int' = 500,
    keys: 'list[str] | None' = None,
    x_axis: 'str' = '_step',
    format: "Literal['default', 'pandas', 'polars']" = 'default',
    stream: "Literal['default', 'system']" = 'default'
)
```

フィルター条件に合致するすべての run について、サンプリングされた履歴メトリクスを返します。 



**引数:**
 
 - `samples`:  各 run で返すサンプル数 
 - `keys`:  特定の キー のメトリクスのみを返す 
 - `x_axis`:  このメトリクスを xAxis に使用（デフォルトは _step） 
 - `format`:  返却形式。"default"、"pandas"、"polars" から選択 
 - `stream`:  メトリクスには "default"、マシンメトリクスには "system" 

**戻り値:**
 
 - `pandas.DataFrame`:  `format="pandas"` の場合、履歴メトリクスの `pandas.DataFrame`。 
 - `polars.DataFrame`:  `format="polars"` の場合、履歴メトリクスの `polars.DataFrame`。 
 - `list of dicts`:  `format="default"` の場合、`run_id` キー を含む履歴メトリクスの dict のリスト。 


---

## <kbd>class</kbd> `Run`
entity と project に関連付けられた 1 つの run。 



**引数:**
 
 - `client`:  W&B API クライアント。 
 - `entity`:  run に関連付けられた entity。 
 - `project`:  run に関連付けられた project。 
 - `run_id`:  run の一意な識別子。 
 - `attrs`:  run の属性。 
 - `include_sweeps`:  この run に Sweeps を含めるかどうか。 



**属性:**
 
 - `tags` ([str]):  run に関連付けられたタグのリスト 
 - `url` (str):  この run の URL 
 - `id` (str):  run の一意な識別子（デフォルトでは 8 文字） 
 - `name` (str):  run の名前 
 - `state` (str):  次のいずれか: running, finished, crashed, killed, preempting, preempted 
 - `config` (dict):  run に関連する ハイパーパラメーター の 辞書 
 - `created_at` (str):  run が開始された時刻の ISO タイムスタンプ 
 - `system_metrics` (dict):  run で最後に記録された システム メトリクス 
 - `summary` (dict):  現在のサマリーを保持する変更可能な 辞書 風プロパティ。update を呼び出すと変更が永続化されます。 
 - `project` (str):  run に関連付けられた project 
 - `entity` (str):  run に関連付けられた entity の名前 
 - `project_internal_id` (int):  project の内部 id 
 - `user` (str):  run を作成した User の名前 
 - `path` (str):  一意の識別子 [entity]/[project]/[run_id] 
 - `notes` (str):  run に関するメモ 
 - `read_only` (boolean):  run が編集可能かどうか 
 - `history_keys` (str):  ログ 済みの履歴メトリクスの キー 
 - `with `wandb.log({key`:  value})` 
 - `metadata` (str):  wandb-metadata.json からの run に関する メタデータ 

### <kbd>method</kbd> `Run.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: 'str',
    project: 'str',
    run_id: 'str',
    attrs: 'Mapping | None' = None,
    include_sweeps: 'bool' = True
)
```

Run オブジェクトを初期化します。 

Run は常に、wandb.Api のインスタンスである api から api.runs() を呼び出すことで初期化されます。 


---

### <kbd>property</kbd> Run.entity

run に関連付けられた entity。 

---

### <kbd>property</kbd> Run.id

run の一意な識別子。 

---


### <kbd>property</kbd> Run.lastHistoryStep

run の履歴で最後に記録された step を返します。 

---

### <kbd>property</kbd> Run.metadata

wandb-metadata.json にある run の メタデータ。 

この メタデータ には、run の説明、タグ、開始時刻、メモリ使用量などが含まれます。 

---

### <kbd>property</kbd> Run.name

run の名前。 

---

### <kbd>property</kbd> Run.path

run のパス。entity、project、run_id を含むリストです。 

---

### <kbd>property</kbd> Run.state

run の状態。Finished、Failed、Crashed、Running のいずれか。 

---

### <kbd>property</kbd> Run.storage_id

run の一意なストレージ識別子。 

---

### <kbd>property</kbd> Run.summary

run に関連付けられたサマリー値を保持する、変更可能な 辞書 風プロパティ。 

---

### <kbd>property</kbd> Run.url

run の URL。 

run の URL は entity、project、run_id から生成されます。SaaS の利用者の場合、形式は `https://wandb.ai/entity/project/run_id` になります。 

---

### <kbd>property</kbd> Run.username

この API は非推奨です。代わりに `entity` を使用してください。 



---

### <kbd>classmethod</kbd> `Run.create`

```python
create(
    api: 'public.Api',
    run_id: 'str | None' = None,
    project: 'str | None' = None,
    entity: 'str | None' = None,
    state: "Literal['running', 'pending']" = 'running'
)
```

指定した project に run を作成します。 

---

### <kbd>method</kbd> `Run.delete`

```python
delete(delete_artifacts=False)
```

指定した run を W&B バックエンドから削除します。 



**引数:**
 
 - `delete_artifacts` (bool, optional):  run に関連付けられた Artifacts を削除するかどうか。 

---

### <kbd>method</kbd> `Run.file`

```python
file(name)
```

Artifact 内で指定した名前のファイルのパスを返します。 



**引数:**
 
 - `name` (str):  取得したいファイル名。 



**戻り値:**
 name 引数 に一致する `File`。 

---

### <kbd>method</kbd> `Run.files`

```python
files(
    names: 'list[str] | None' = None,
    pattern: 'str | None' = None,
    per_page: 'int' = 50
)
```

条件に一致する、この run 内のすべてのファイルに対する `Files` オブジェクトを返します。 

一致させたいファイル名のリスト、またはパターンを指定できます。両方を指定した場合、pattern は無視されます。 



**引数:**
 
 - `names` (list):  取得したいファイル名。空ならすべてのファイルを返す 
 - `pattern` (str, optional):  W&B からファイルを返す際に使用するパターン。MySQL の LIKE 構文を使用します。例えば .json で終わるすべてのファイルにマッチさせるには "%.json"。names と pattern の両方を指定すると ValueError が送出されます。 
 - `per_page` (int):  1 ページあたりの結果数。 



**戻り値:**
 `File` オブジェクトを反復する `Files` オブジェクト。 

---

### <kbd>method</kbd> `Run.history`

```python
history(samples=500, keys=None, x_axis='_step', pandas=True, stream='default')
```

1 つの run に対してサンプリングされた履歴メトリクスを返します。 

履歴レコードがサンプリングされていても問題なければ、こちらのほうがシンプルかつ高速です。 



**引数:**
 
 - `samples `:  (int, optional) 返すサンプル数 
 - `pandas `:  (bool, optional) pandas DataFrame を返すかどうか 
 - `keys `:  (list, optional) 特定の キー のメトリクスのみを返す 
 - `x_axis `:  (str, optional) このメトリクスを xAxis に使用（デフォルトは _step） 
 - `stream `:  (str, optional) メトリクスには "default"、マシンメトリクスには "system" 



**戻り値:**
 
 - `pandas.DataFrame`:  pandas=True の場合、履歴メトリクスの `pandas.DataFrame`。 
 - `list of dicts`:  pandas=False の場合、履歴メトリクスの dict のリスト。 

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
    aliases: 'Collection[str] | None' = None,
    tags: 'Collection[str] | None' = None
)
```

Artifact を run の出力として宣言します。 



**引数:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)` から返された Artifact。 
 - `aliases` (list, optional):  この Artifact に適用するエイリアス。 
 - `tags`:  (list, optional) この Artifact に適用するタグ（任意）。 



**戻り値:**
 `Artifact` オブジェクト。 

---

### <kbd>method</kbd> `Run.logged_artifacts`

```python
logged_artifacts(per_page: 'int' = 100) → public.RunArtifacts
```

この run によって ログ されたすべての Artifacts を取得します。 

run 中に ログ されたすべての出力 Artifact を取得します。反復可能なページネーション結果として返され、イテレーションまたは単一のリストに収集できます。 



**引数:**
 
 - `per_page`:  1 回の API リクエストで取得する Artifact 数。 



**戻り値:**
 この run で出力として ログ されたすべての Artifact オブジェクトの反復可能コレクション。 



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

run オブジェクトへの変更を W&B バックエンドに永続化します。 

---

### <kbd>method</kbd> `Run.scan_history`

```python
scan_history(keys=None, page_size=1000, min_step=None, max_step=None)
```

1 つの run について、すべての履歴レコードを反復可能コレクションとして返します。 



**引数:**
 
 - `keys` ([str], optional):  これらの キー のみを取得し、かつすべての キー が定義された行のみを取得。 
 - `page_size` (int, optional):  API から取得するページのサイズ。 
 - `min_step` (int, optional):  一度にスキャンする最小ステップ数。 
 - `max_step` (int, optional):  一度にスキャンする最大ステップ数。 



**戻り値:**
 履歴レコード（dict）に対する反復可能コレクション。 



**例:**
 サンプル run の loss 値をすべてエクスポートします。 

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

この run を表示する iframe を含む HTML を生成します。 

---

### <kbd>method</kbd> `Run.update`

```python
update()
```

run オブジェクトへの変更を W&B バックエンドに永続化します。 

---

### <kbd>method</kbd> `Run.upload_file`

```python
upload_file(path, root='.')
```

ローカルファイルを W&B にアップロードし、この run に関連付けます。 



**引数:**
 
 - `path` (str):  アップロードするファイルのパス。絶対パスまたは相対パスを指定可能。 
 - `root` (str):  ファイルを保存する際の基準パス。例えば、run 内で "my_dir/file.txt" として保存したく、現在の作業ディレクトリーが "my_dir" の場合は "../" を指定します。デフォルトは現在のディレクトリー（"."）。 



**戻り値:**
 アップロードされたファイルを表す `File` オブジェクト。 

---

### <kbd>method</kbd> `Run.use_artifact`

```python
use_artifact(artifact, use_as=None)
```

Artifact を run の入力として宣言します。 



**引数:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)` から返された Artifact 
 - `use_as` (string, optional):  スクリプト内で Artifact をどのように使用するかを識別する文字列。ベータ版の wandb Launch 機能の Artifact スワップ機能を使用する際、run で使用した Artifact を簡単に区別するために使われます。 



**戻り値:**
 `Artifact` オブジェクト。 

---

### <kbd>method</kbd> `Run.used_artifacts`

```python
used_artifacts(per_page: 'int' = 100) → public.RunArtifacts
```

この run で明示的に使用された Artifacts を取得します。 

通常は `run.use_artifact()` を通じて run 中に明示的に入力として宣言された Artifact のみを取得します。反復可能なページネーション結果として返され、イテレーションまたは単一のリストに収集できます。 



**引数:**
 
 - `per_page`:  1 回の API リクエストで取得する Artifact 数。 



**戻り値:**
 この run で入力として明示的に使用された Artifact オブジェクトの反復可能コレクション。 



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

run が終了するまで状態を確認します。