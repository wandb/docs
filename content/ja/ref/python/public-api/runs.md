---
title: run
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-runs
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Runs。

このモジュールは、W&B の Run およびそれに関連するデータとやりとりするためのクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# フィルターに一致する Run を取得
runs = Api().runs(
     path="entity/project", filters={"state": "finished", "config.batch_size": 32}
)

# Run のデータにアクセス
for run in runs:
     print(f"Run: {run.name}")
     print(f"Config: {run.config}")
     print(f"Metrics: {run.summary}")

     # pandas で履歴を取得
     history_df = run.history(keys=["loss", "accuracy"], pandas=True)

     # アーティファクトの操作
     for artifact in run.logged_artifacts():
         print(f"Artifact: {artifact.name}")
``` 



**注:**

> このモジュールは W&B Public API の一部であり、run データへの読み書きアクセスを提供します。新しい run をロギングするには、wandb パッケージの wandb.init() 関数を使用してください。

## <kbd>class</kbd> `Runs`
プロジェクトに紐づき、オプションでフィルターされた Run のイテラブルなコレクション。

通常は `Api.runs` 名前空間経由で間接的に利用されます。



**引数:**
 
 - `client`:  (`wandb.apis.public.RetryingClient`) リクエストに使用する API クライアント
 - `entity`:  (str) プロジェクトの所有者（ユーザー名またはチーム）
 - `project`:  (str) run を取得するプロジェクト名
 - `filters`:  (Optional[Dict[str, Any]]) run クエリに適用するフィルターの辞書
 - `order`:  (Optional[str]) run の並び順。"asc" または "desc"、デフォルトは "desc"
 - `per_page`:  (int) リクエストごとにフェッチする run の数（デフォルトは 50）
 - `include_sweeps`:  (bool) run に sweep 情報を含めるかどうか。デフォルトは True



**使用例:**
 ```python
from wandb.apis.public.runs import Runs
from wandb.apis.public import Api

# フィルターを満たすプロジェクト内すべての run を取得
filters = {"state": "finished", "config.optimizer": "adam"}

runs = Api().runs(
    client=api.client,
    entity="entity",
    project="project_name",
    filters=filters,
)

# run ごとに詳細を表示
for run in runs:
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")
    print(f"Run state: {run.state}")
    print(f"Run config: {run.config}")
    print(f"Run summary: {run.summary}")
    print(f"Run history (samples=5): {run.history(samples=5)}")
    print("----------")

# 特定のメトリクスを持つすべての run の履歴を取得
histories_df = runs.histories(
    samples=100,  # 各 run のサンプル数
    keys=["loss", "accuracy"],  # 取得するメトリクス
    x_axis="_step",  # X軸のメトリクス
    format="pandas",  # pandas DataFrame で返す
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

フィルター条件に合致するすべての run について、サンプリングされた履歴メトリクスを返します。



**引数:**
 
 - `samples`:  各 run ごとに返すサンプル数
 - `keys`:  特定のキーのメトリクスのみを返す
 - `x_axis`:  このメトリクスを X軸として使用（デフォルトは _step）
 - `format`:  データの返却フォーマット。"default"、"pandas"、"polars" が利用可能
 - `stream`:  メトリクスなら "default"、マシンメトリクスなら "system"

**返り値:**
 
 - `pandas.DataFrame`:  `format="pandas"` の場合、履歴メトリクスの `pandas.DataFrame` を返します
 - `polars.DataFrame`:  `format="polars"` の場合、履歴メトリクスの `polars.DataFrame` を返します
 - `list of dicts`:  `format="default"` の場合、各履歴メトリクスと `run_id` キーを含む辞書リストを返します


---

## <kbd>class</kbd> `Run`
特定の entity と project に紐づく 1 つの run。



**引数:**
 
 - `client`:  W&B API クライアント
 - `entity`:  run が属する entity
 - `project`:  run が属するプロジェクト
 - `run_id`:  一意な run 識別子
 - `attrs`:  run の属性
 - `include_sweeps`:  sweeps を含めるかどうか



**属性:**
 
 - `tags` ([str]):  run に関連付けられたタグのリスト
 - `url` (str):  この run の URL
 - `id` (str):  run の一意な ID（デフォルトは8文字）
 - `name` (str):  run の名前
 - `state` (str):  状態（running, finished, crashed, killed, preempting, preempted のいずれか）
 - `config` (dict):  run に関連付けられたハイパーパラメータの辞書
 - `created_at` (str):  run 開始時の ISO タイムスタンプ
 - `system_metrics` (dict):  run で記録された最新のシステムメトリクス
 - `summary` (dict):  現在のサマリ情報を保持する、ミュータブルな辞書的プロパティ。update を呼ぶことで変更を保存可能
 - `project` (str):  run が属するプロジェクト
 - `entity` (str):  run が属する entity の名前
 - `project_internal_id` (int):  プロジェクトの内部 ID
 - `user` (str):  run を作成したユーザーの名前
 - `path` (str):  一意な識別子 [entity]/[project]/[run_id]
 - `notes` (str):  run に関するノート
 - `read_only` (boolean):  run の編集可否
 - `history_keys` (str):  ログされた履歴メトリクスのキー
 - `with wandb.log({key`:  value})` 
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

Run は常に wandb.Api インスタンスの api.runs() を呼ぶことで初期化されます。


---

### <kbd>property</kbd> Run.entity

run に関連付けられた entity です。

---

### <kbd>property</kbd> Run.id

run の一意な識別子です。

---


### <kbd>property</kbd> Run.lastHistoryStep

run の履歴で最後に記録されたステップを返します。

---

### <kbd>property</kbd> Run.metadata

wandb-metadata.json から取得される run のメタデータ。

内容には run の説明、タグ、開始時刻、メモリ使用量などが含まれます。

---

### <kbd>property</kbd> Run.name

run の名前です。

---

### <kbd>property</kbd> Run.path

run のパスです。パスは entity, project, run_id のリストです。

---

### <kbd>property</kbd> Run.state

run の状態です。Finished、Failed、Crashed、Running のいずれか。

---

### <kbd>property</kbd> Run.storage_id

run の一意なストレージ識別子です。

---

### <kbd>property</kbd> Run.summary

run に紐づくサマリ値を保持する、ミュータブルな辞書的プロパティです。

---

### <kbd>property</kbd> Run.url

run の URL です。

run URL は entity, project, run_id から生成されます。SaaS ユーザーでは `https://wandb.ai/entity/project/run_id` 形式になります。

---

### <kbd>property</kbd> Run.username

この API は非推奨です。代わりに `entity` を使用してください。



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

wandb バックエンドから指定の run を削除します。



**引数:**
 
 - `delete_artifacts` (bool, optional):  run に関連付けられたアーティファクトを削除するかどうか

---

### <kbd>method</kbd> `Run.file`

```python
file(name)
```

アーティファクト内で指定された名前のファイルのパスを返します。



**引数:**
 
 - `name` (str):  取得したいファイル名



**戻り値:**
 name 引数に一致する `File` オブジェクト

---

### <kbd>method</kbd> `Run.files`

```python
files(names=None, per_page=50)
```

指定した名前の各ファイルについてファイルパスを返します。



**引数:**
 
 - `names` (list):  取得したいファイル名。空の場合はすべてのファイルを返します
 - `per_page` (int):  1ページあたりの結果数



**戻り値:**
 `Files` オブジェクト。`File` オブジェクトをイテレーション可能なもの

---

### <kbd>method</kbd> `Run.history`

```python
history(samples=500, keys=None, x_axis='_step', pandas=True, stream='default')
```

run の履歴メトリクスをサンプリングして返します。

履歴レコードがサンプリングされても構わない場合、より簡単かつ高速に利用できます。



**引数:**
 
 - `samples `:  (int, optional) 返すサンプル数
 - `pandas `:  (bool, optional) pandas dataframe で返すか
 - `keys `:  (list, optional) 指定したキーのメトリクスのみ返す
 - `x_axis `:  (str, optional) このメトリクスを x 軸として使用（デフォルトは _step）
 - `stream `:  (str, optional) メトリクスなら "default"、マシンメトリクスなら "system"



**返り値:**
 
 - `pandas.DataFrame`:  pandas=True の場合は履歴メトリクスの `pandas.DataFrame`
 - `list of dicts`:  pandas=False の場合は履歴メトリクスの辞書リスト

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

この run の出力としてアーティファクトを宣言します。



**引数:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)` から返されたアーティファクト
 - `aliases` (list, optional):  このアーティファクトに適用するエイリアス
 - `tags`:  (list, optional) このアーティファクトにつけるタグ（任意）



**戻り値:**
 `Artifact` オブジェクト

---

### <kbd>method</kbd> `Run.logged_artifacts`

```python
logged_artifacts(per_page: int = 100) → RunArtifacts
```

この run でログされたすべてのアーティファクトを取得します。

run 中に出力としてログされたすべてのアーティファクトを取得します。ページネーションされた結果として返され、イテレーションまたはリストへのまとめが可能です。



**引数:**
 
 - `per_page`:  1回の API リクエストで取得するアーティファクト数



**戻り値:**
 この run 中に出力としてログされたすべての Artifact オブジェクトのイテラブルなコレクション



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
 
 - `keys` ([str], optional):  これらのキーのみを取得し、すべてのキーが定義された行のみ取得
 - `page_size` (int, optional):  API から取得するページサイズ
 - `min_step` (int, optional):  一度にスキャンする最小ページ数
 - `max_step` (int, optional):  一度にスキャンする最大ページ数



**戻り値:**
 履歴レコード（辞書）のイテラブルコレクション



**例:**
 loss 値をすべてエクスポートする例

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

run オブジェクトへの変更を wandb バックエンドに保存します。

---

### <kbd>method</kbd> `Run.upload_file`

```python
upload_file(path, root='.')
```

ローカルファイルを W&B にアップロードし、この run に紐づけます。



**引数:**
 
 - `path` (str):  アップロードするファイルのパス（絶対または相対パス可）
 - `root` (str):  ファイルをどのパスで保存するかの基点パス。例えば今 "my_dir" にいて "my_dir/file.txt" を run に保存したい場合は root を "../" にします。デフォルトは現在のディレクトリ(".")



**戻り値:**
 アップロードされたファイルを表現する `File` オブジェクト

---

### <kbd>method</kbd> `Run.use_artifact`

```python
use_artifact(artifact, use_as=None)
```

アーティファクトを run の入力として宣言します。



**引数:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)` から返されたアーティファクト
 - `use_as` (string, optional):  スクリプト内でアーティファクトをどのように使うかを示す識別子。beta バージョンの wandb launch 機能のアーティファクトスワッピング時の判別用



**戻り値:**
 `Artifact` オブジェクト

---

### <kbd>method</kbd> `Run.used_artifacts`

```python
used_artifacts(per_page: int = 100) → RunArtifacts
```

この run で明示的に使用されたアーティファクトを取得します。

run 中で明示的に use_artifact() された入力アーティファクトのみを取得します。ページネーションされた結果として返され、イテレーションまたはリストへのまとめが可能です。



**引数:**
 
 - `per_page`:  1回の API リクエストで取得するアーティファクト数



**戻り値:**
 この run で入力として明示的に使用された Artifact オブジェクトのイテラブルなコレクション



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

run の状態が終了するまで監視します。