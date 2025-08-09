---
title: プロジェクト
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-projects
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API の Project オブジェクト用モジュールです。

このモジュールは、W&B Projects とそれに関連するデータを操作するためのクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# エンティティに紐づくすべての Project を取得
projects = Api().projects("entity")

# Project のデータへアクセス
for project in projects:
     print(f"Project: {project.name}")
     print(f"URL: {project.url}")

     # Artifact タイプを取得
     for artifact_type in project.artifacts_types():
         print(f"Artifact Type: {artifact_type.name}")

     # Sweep を取得
     for sweep in project.sweeps():
         print(f"Sweep ID: {sweep.id}")
         print(f"State: {sweep.state}")
```



**補足:**

> このモジュールは W&B Public API の一部であり、Project へのアクセスや管理のためのメソッドを提供します。新しい Project を作成するには、wandb.init() で新しいプロジェクト名を指定してください。

## <kbd>class</kbd> `Projects`
`Project` オブジェクトのイテラブルコレクションです。

エンティティによって作成・保存された Projects へのイテラブルなインターフェースを提供します。



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用する API クライアントインスタンス。
 - `entity` (str):  Projects を取得したいエンティティ名（ユーザー名またはチーム名）。
 - `per_page` (int):  1リクエストあたり取得する Projects の件数（デフォルトは50）。



**例:**
 ```python
from wandb.apis.public.api import Api

# エンティティに紐づく Project を取得
projects = Api().projects(entity="entity")

# Project ごとに繰り返し処理
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

`Project` オブジェクトのイテラブルコレクション。



**引数:**
 
 - `client`:  W&B へクエリを投げるための API クライアント。
 - `entity`:  Projects を管理するエンティティ。
 - `per_page`:  API リクエストごとに取得する Projects の数。


---





## <kbd>class</kbd> `Project`
Project は run の名前空間です。



**引数:**
 
 - `client`:  W&B API クライアントインスタンス。
 - `name` (str):  Project の名前。
 - `entity` (str):  Project を所有するエンティティ名。

### <kbd>method</kbd> `Project.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    attrs: dict
) → Project
```

エンティティに紐づく 1 つの Project を表します。



**引数:**
 
 - `client`:  W&B へクエリを投げるための API クライアント。
 - `entity`:  Project を所有するエンティティ。
 - `project`:  データ取得対象となる Project 名。
 - `attrs`:  Project の各属性。


---

### <kbd>property</kbd> Project.id





---

### <kbd>property</kbd> Project.path

Project のパスを返します。パスは entity と project 名から成るリストです。

---

### <kbd>property</kbd> Project.url

Project の URL を返します。



---

### <kbd>method</kbd> `Project.artifacts_types`

```python
artifacts_types(per_page=50)
```

この Project に紐づくすべての Artifact タイプを返します。

---

### <kbd>method</kbd> `Project.sweeps`

```python
sweeps()
```

Project に関連づけられたすべての Sweep を取得します。

---