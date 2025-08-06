---
title: プロジェクト
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API による Project オブジェクトの操作

このモジュールは、W&B の Projects や関連データとやりとりするクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# エンティティに紐づくすべてのプロジェクトを取得
projects = Api().projects("entity")

# プロジェクトデータへアクセス
for project in projects:
     print(f"Project: {project.name}")
     print(f"URL: {project.url}")

     # アーティファクトタイプを取得
     for artifact_type in project.artifacts_types():
         print(f"Artifact Type: {artifact_type.name}")

     # スイープを取得
     for sweep in project.sweeps():
         print(f"Sweep ID: {sweep.id}")
         print(f"State: {sweep.state}")
```



**注意:**

> このモジュールは W&B Public API の一部であり、Projects へのアクセスや管理のためのメソッドを提供します。新しいプロジェクトを作成する場合は、wandb.init() で新規プロジェクト名を指定してください。

## <kbd>class</kbd> `Projects`
`Project` オブジェクトのイテラブルなコレクション

エンティティによって作成・保存された Projects へアクセスするためのイテラブルなインターフェースです。



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用する API クライアントインスタンス 
 - `entity` (str):  Projects を取得したいエンティティ名（ユーザー名またはチーム名） 
 - `per_page` (int):  1リクエストあたり取得する Projects の数（デフォルト: 50）



**例:**
 ```python
from wandb.apis.public.api import Api

# 指定したエンティティに属するプロジェクトを取得
projects = Api().projects(entity="entity")

# プロジェクト情報を表示
for project in projects:
    print(f"Project: {project.name}")
    print(f"- URL: {project.url}")
    print(f"- Created at: {project.created_at}")
    print(f"- Is benchmark: {project.is_benchmark}")
```

### <kbd>method</kbd> `Projects.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    per_page: int = 50
) → Projects
```

`Project` オブジェクトのイテラブルなコレクションです。



**引数:**
 
 - `client`:  W&B への問い合わせで用いる API クライアント
 - `entity`:  Projects の所有エンティティ
 - `per_page`:  API から1回で取得する Projects 数


---





## <kbd>class</kbd> `Project`
プロジェクトは Run の名前空間です。



**引数:**
 
 - `client`:  W&B API クライアントインスタンス
 - `name` (str):  プロジェクト名
 - `entity` (str):  プロジェクトを所有するエンティティ名

### <kbd>method</kbd> `Project.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    attrs: dict
) → Project
```

エンティティと紐づけられた1つの Project を表します。



**引数:**
 
 - `client`:  W&B への問い合わせで用いる API クライアント
 - `entity`:  Projects の所有エンティティ
 - `project`:  取得したいプロジェクト名
 - `attrs`:  プロジェクトの属性情報


---

### <kbd>property</kbd> Project.id





---

### <kbd>property</kbd> Project.path

プロジェクトのパスを返します。パスは entity 名と project 名からなるリストです。

---

### <kbd>property</kbd> Project.url

プロジェクトの URL を返します。



---

### <kbd>method</kbd> `Project.artifacts_types`

```python
artifacts_types(per_page=50)
```

このプロジェクトに関連付けられているすべてのアーティファクトタイプを返します。

---

### <kbd>method</kbd> `Project.sweeps`

```python
sweeps()
```

このプロジェクトに紐づくすべてのスイープ情報を取得します。

---