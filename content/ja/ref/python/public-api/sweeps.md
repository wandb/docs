---
title: sweeps
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-sweeps
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Sweeps. 

このモジュールは、W&B のハイパーパラメーター最適化 Sweeps と対話するためのクラスを提供します。 



**Example:**
 ```python
from wandb.apis.public import Api

# 特定の sweep を取得
sweep = Api().sweep("entity/project/sweep_id")

# sweep のプロパティにアクセス
print(f"Sweep: {sweep.name}")
print(f"State: {sweep.state}")
print(f"Best Loss: {sweep.best_loss}")

# 最良の run を取得
best_run = sweep.best_run()
print(f"Best Run: {best_run.name}")
print(f"Metrics: {best_run.summary}")
``` 



**Note:**

> このモジュールは W&B Public API の一部で、sweep データへの読み取り専用のアクセスを提供します。sweep の作成や制御には、メインの wandb パッケージにある wandb.sweep() と wandb.agent() を使用してください。 

## <kbd>class</kbd> `Sweeps`
`Sweep` オブジェクトのコレクションに対する遅延イテレーター。 



**Examples:**
 ```python
from wandb.apis.public import Api

sweeps = Api().project(name="project_name", entity="entity").sweeps()

# sweep を反復して詳細を出力
for sweep in sweeps:
     print(f"Sweep name: {sweep.name}")
     print(f"Sweep ID: {sweep.id}")
     print(f"Sweep URL: {sweep.url}")
     print("----------")
``` 

### <kbd>method</kbd> `Sweeps.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    per_page: int = 50
) → Sweeps
```

`Sweep` オブジェクトの反復可能なコレクション。 



**Args:**
 
 - `client`:  W&B にクエリを送るための API クライアント。 
 - `entity`:  Sweeps を所有する entity。 
 - `project`:  Sweeps を含むプロジェクト。 
 - `per_page`:  API へのリクエストごとに取得する sweep の件数。 


---


### <kbd>property</kbd> Sweeps.length





---




## <kbd>class</kbd> `Sweep`
sweep に関連付けられた run の集合。 



**Attributes:**
 
 - `runs` (Runs):  run のリスト 
 - `id` (str):  Sweep ID 
 - `project` (str):  sweep が属するプロジェクト名 
 - `config` (dict):  sweep configuration を含む辞書 
 - `state` (str):  sweep の状態。"Finished"、"Failed"、"Crashed"、"Running" のいずれか。 
 - `expected_run_count` (int):  sweep に対して期待される run の数 

### <kbd>method</kbd> `Sweep.__init__`

```python
__init__(client, entity, project, sweep_id, attrs=None)
```






---

### <kbd>property</kbd> Sweep.config

sweep で使用される sweep configuration。 

---

### <kbd>property</kbd> Sweep.entity

sweep に関連付けられている entity。 

---

### <kbd>property</kbd> Sweep.expected_run_count

sweep における期待される run の数を返します。無限の run の場合は None。 

---

### <kbd>property</kbd> Sweep.name

sweep の名前。 

次の優先順位で、最初に存在する名前を返します: 

1. ユーザーが編集した表示名 2. 作成時に設定された名前 3. Sweep ID 

---

### <kbd>property</kbd> Sweep.order

sweep の order キーを返します。 

---

### <kbd>property</kbd> Sweep.path

プロジェクトのパスを返します。 

パスは entity、プロジェクト名、sweep ID からなるリストです。 

---

### <kbd>property</kbd> Sweep.url

sweep の URL。 

sweep の URL は entity、project、語句 "sweeps"、そして sweep ID.run_id から生成されます。SaaS ユーザーの場合は `https://wandb.ai/entity/project/sweeps/sweeps_ID` の形式になります。 

---

### <kbd>property</kbd> Sweep.username

非推奨です。代わりに `Sweep.entity` を使用してください。 



---

### <kbd>method</kbd> `Sweep.best_run`

```python
best_run(order=None)
```

config で定義されたメトリクス、または指定した order に従って並べ替えた最良の run を返します。 

---

### <kbd>classmethod</kbd> `Sweep.get`

```python
get(
    client: 'RetryingClient',
    entity: Optional[str] = None,
    project: Optional[str] = None,
    sid: Optional[str] = None,
    order: Optional[str] = None,
    query: Optional[str] = None,
    **kwargs
)
```

クラウド バックエンドに対してクエリを実行します。 



**Args:**
 
 - `client`:  クエリの実行に使用するクライアント。 
 - `entity`:  プロジェクトの所有者 (ユーザー名またはチーム)。 
 - `project`:  sweep を取得するプロジェクト名。 
 - `sid`:  クエリ対象の sweep ID。 
 - `order`:  sweep の run を返す順序。 
 - `query`:  実行に使用するクエリ。 
 - `**kwargs`:  クエリに渡す追加のキーワード引数。 

---


### <kbd>method</kbd> `Sweep.to_html`

```python
to_html(height=420, hidden=False)
```

この sweep を表示する iframe を含む HTML を生成します。