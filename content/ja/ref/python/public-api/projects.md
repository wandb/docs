---
title: projects
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-projects
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Project オブジェクト向けの W&B Public API。 

このモジュールは、W&B の Projects とそれに関連するデータとやり取りするためのクラスを提供します。 



**例:**
 ```python
from wandb.apis.public import Api

# 指定した Entity の Projects をすべて取得
projects = Api().projects("entity")

# Project のデータにアクセス
for project in projects:
     print(f"Project: {project.name}")
     print(f"URL: {project.url}")

     # Artifact Type を取得
     for artifact_type in project.artifacts_types():
         print(f"Artifact Type: {artifact_type.name}")

     # Sweeps を取得
     for sweep in project.sweeps():
         print(f"Sweep ID: {sweep.id}")
         print(f"State: {sweep.state}")
``` 



**注記:**

> このモジュールは W&B の Public API の一部で、Projects へのアクセスと管理のためのメソッドを提供します。新しい Project を作成するには、wandb.init() を新しい Project 名とともに使用してください。 

## <kbd>class</kbd> `Projects`
`Project` オブジェクトの遅延イテレータ。 

Entity によって作成・保存された Projects にアクセスするための反復可能なインターフェース。 



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用する API クライアント インスタンス。 
 - `entity` (str):  Projects を取得する対象の Entity 名（ユーザー名または Team）。 
 - `per_page` (int):  1 回のリクエストで取得する Project 数（デフォルトは 50）。 



**例:**
 ```python
from wandb.apis.public.api import Api

# この Entity に属する Projects を探す
projects = Api().projects(entity="entity")

# Projects を反復処理する
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

`Project` オブジェクトの反復可能なコレクション。 



**引数:**
 
 - `client`:  W&B にクエリを送るために使用する API クライアント。 
 - `entity`:  Projects を所有する Entity。 
 - `per_page`:  API への 1 回のリクエストで取得する Project 数。 


---





## <kbd>class</kbd> `Project`
Project は Runs のためのネームスペースです。 



**引数:**
 
 - `client`:  W&B の API クライアント インスタンス。 
 - `name` (str):  Project の名前。 
 - `entity` (str):  その Project を所有する Entity 名。 

### <kbd>method</kbd> `Project.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    attrs: dict
) → Project
```

ある Entity に関連付けられた単一の Project。 



**引数:**
 
 - `client`:  W&B にクエリを送るために使用する API クライアント。 
 - `entity`:  Projects を所有する Entity。 
 - `project`:  クエリ対象の Project 名。 
 - `attrs`:  Project の属性。 


---

### <kbd>property</kbd> Project.id





---

### <kbd>property</kbd> Project.path

Project のパスを返します。パスは Entity と Project 名を含むリストです。 

---

### <kbd>property</kbd> Project.url

Project の URL を返します。 



---

### <kbd>method</kbd> `Project.artifacts_types`

```python
artifacts_types(per_page=50)
```

この Project に関連付けられたすべての Artifact Type を返します。 

---

### <kbd>method</kbd> `Project.sweeps`

```python
sweeps(per_page=50)
```

この Project 内の Sweeps のページングされたコレクションを返します。 



**引数:**
 
 - `per_page`:  API への 1 回のリクエストで取得する Sweep 数。 



**戻り値:**
 `Sweep` オブジェクトの反復可能なコレクションである `Sweeps` オブジェクト。 

---