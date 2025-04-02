---
title: Api
menu:
  reference:
    identifier: ja-ref-python-public-api-api
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L104-L1573 >}}

wandb サーバー のクエリに使用されます。

```python
Api(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) -> None
```

#### 例:

初期化の最も一般的な方法

```
>>> wandb.Api()
```

| Args |  |
| :--- | :--- |
|  `overrides` |  (辞書) `https://api.wandb.ai` 以外の wandb サーバー を使用している場合は、`base_url` を設定できます。また、`entity`、`project`、および `run` のデフォルト値を設定することもできます。 |

| Attributes |  |
| :--- | :--- |

## メソッド

### `artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1299-L1321)

```python
artifact(
    name: str,
    type: Optional[str] = None
)
```

`project/name` または `entity/project/name` の形式でパスを解析して、単一の artifact を返します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) artifact 名。project/ または entity/project/ が前に付いている場合があります。名前で entity が指定されていない場合、Run または API 設定の entity が使用されます。有効な名前は、name:version name:alias の形式にすることができます。 |
|  `type` |  (str, オプション) フェッチする artifact の型。 |

| Returns |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  artifact 名が指定されていない場合。 |
|  `ValueError` |  artifact の型が指定されているが、フェッチされた artifact の型と一致しない場合。 |

#### 注:

このメソッドは、外部での使用のみを目的としています。wandb リポジトリ コード内で `api.artifact()` を呼び出さないでください。

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1181-L1210)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

型で単一の artifact collection を返し、`entity/project/name` の形式でパスを解析します。

| Args |  |
| :--- | :--- |
|  `type_name` |  (str) フェッチする artifact collection の型。 |
|  `name` |  (str) artifact collection 名。entity/project が前に付いている場合があります。 |

| Returns |  |
| :--- | :--- |
|  `ArtifactCollection` オブジェクト。 |

### `artifact_collection_exists`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1442-L1459)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

artifact collection が指定された project および entity 内に存在するかどうかを返します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) artifact collection 名。entity/project が前に付いている場合があります。entity または project が指定されていない場合、オーバーライド パラメータから推測されます (設定されている場合)。それ以外の場合、entity はユーザー設定から取得され、project はデフォルトで "uncategorized" になります。 |
|  `type` |  (str) artifact collection の型 |

| Returns |  |
| :--- | :--- |
|  artifact collection が存在する場合は True、それ以外の場合は False。 |

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1154-L1179)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) -> "public.ArtifactCollections"
```

一致する artifact collection の collection を返します。

| Args |  |
| :--- | :--- |
|  `project_name` |  (str) フィルタリングする project の名前。 |
|  `type_name` |  (str) フィルタリングする artifact の型の名前。 |
|  `per_page` |  (int, オプション) クエリ ページネーションのページ サイズを設定します。None を指定すると、デフォルト サイズが使用されます。通常、これを変更する理由はありません。 |

| Returns |  |
| :--- | :--- |
|  反復可能な `ArtifactCollections` オブジェクト。 |

### `artifact_exists`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1420-L1440)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

artifact バージョンが指定された project および entity 内に存在するかどうかを返します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) artifact 名。entity/project が前に付いている場合があります。entity または project が指定されていない場合、オーバーライド パラメータから推測されます (設定されている場合)。それ以外の場合、entity はユーザー設定から取得され、project はデフォルトで "uncategorized" になります。有効な名前は、name:version name:alias の形式にすることができます。 |
|  `type` |  (str, オプション) artifact の型 |

| Returns |  |
| :--- | :--- |
|  artifact バージョンが存在する場合は True、それ以外の場合は False。 |

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1130-L1152)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

一致する `ArtifactType` を返します。

| Args |  |
| :--- | :--- |
|  `type_name` |  (str) 取得する artifact の型の名前。 |
|  `project` |  (str, オプション) 指定された場合、フィルタリングする project 名またはパス。 |

| Returns |  |
| :--- | :--- |
|  `ArtifactType` オブジェクト。 |

### `artifact_types`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1109-L1128)

```python
artifact_types(
    project: Optional[str] = None
) -> "public.ArtifactTypes"
```

一致する artifact 型の collection を返します。

| Args |  |
| :--- | :--- |
|  `project` |  (str, オプション) 指定された場合、フィルタリングする project 名またはパス。 |

| Returns |  |
| :--- | :--- |
|  反復可能な `ArtifactTypes` オブジェクト。 |

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1212-L1222)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

非推奨。代わりに `artifacts(type_name, name)` を使用してください。

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1224-L1260)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: Optional[int] = 50,
    tags: Optional[List[str]] = None
) -> "public.Artifacts"
```

指定されたパラメータから `Artifacts` collection を返します。

| Args |  |
| :--- | :--- |
|  `type_name` |  (str) フェッチする artifacts の型。 |
|  `name` |  (str) artifact collection 名。entity/project が前に付いている場合があります。 |
|  `per_page` |  (int, オプション) クエリ ページネーションのページ サイズを設定します。None を指定すると、デフォルト サイズが使用されます。通常、これを変更する理由はありません。 |
|  `tags` |  (list[str], オプション) これらのタグをすべて持つ artifacts のみを返します。 |

| Returns |  |
| :--- | :--- |
|  反復可能な `Artifacts` オブジェクト。 |

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L294-L301)

```python
create_project(
    name: str,
    entity: str
) -> None
```

新しい project を作成します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) 新しい project の名前。 |
|  `entity` |  (str) 新しい project の entity。 |

### `create_run`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L303-L323)

```python
create_run(
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) -> "public.Run"
```

新しい run を作成します。

| Args |  |
| :--- | :--- |
|  `run_id` |  (str, オプション) 指定された場合、run に割り当てる ID。run ID はデフォルトで自動的に生成されるため、通常はこれを指定する必要はなく、自己責任で行う必要があります。 |
|  `project` |  (str, オプション) 指定された場合、新しい run の project。 |
|  `entity` |  (str, オプション) 指定された場合、新しい run の entity。 |

| Returns |  |
| :--- | :--- |
|  新しく作成された `Run`。 |

### `create_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L325-L435)

```python
create_run_queue(
    name: str,
    type: "public.RunQueueResourceType",
    entity: Optional[str] = None,
    prioritization_mode: Optional['public.RunQueuePrioritizationMode'] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) -> "public.RunQueue"
```

新しい run queue (Launch) を作成します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) 作成する queue の名前 |
|  `type` |  (str) queue に使用するリソースの型。"local-container"、"local-process"、"kubernetes"、"sagemaker"、または "gcp-vertex" のいずれか。 |
|  `entity` |  (str) queue を作成する entity のオプションの名前。None の場合、設定されたまたはデフォルトの entity が使用されます。 |
|  `prioritization_mode` |  (str) 使用する優先順位付けのオプションのバージョン。"V0" または None |
|  `config` |  (dict) queue に使用するオプションのデフォルト リソース設定。handlebars (例: `{{var}}`) を使用してテンプレート変数を指定します。 |
|  `template_variables` |  (dict) config で使用するテンプレート変数スキーマの辞書。予期される形式: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } }` |

| Returns |  |
| :--- | :--- |
|  新しく作成された `RunQueue` |

| Raises |  |
| :--- | :--- |
|  パラメータが無効な場合は ValueError wandb API エラーの場合は wandb.Error |

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L843-L853)

```python
create_team(
    team, admin_username=None
)
```

新しい team を作成します。

| Args |  |
| :--- | :--- |
|  `team` |  (str) team の名前 |
|  `admin_username` |  (str) team の管理者ユーザーのオプションのユーザー名。デフォルトは現在のユーザーです。 |

| Returns |  |
| :--- | :--- |
|  `Team` オブジェクト |

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L552-L562)

```python
create_user(
    email, admin=(False)
)
```

新しい user を作成します。

| Args |  |
| :--- | :--- |
|  `email` |  (str) ユーザーのメール アドレス |
|  `admin` |  (bool) このユーザーをグローバル インスタンス管理者にするかどうか |

| Returns |  |
| :--- | :--- |
|  `User` オブジェクト |

### `flush`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L629-L636)

```python
flush()
```

ローカル キャッシュをフラッシュします。

api オブジェクトは run のローカル キャッシュを保持するため、スクリプトの実行中に run の状態が変化する可能性がある場合は、`api.flush()` でローカル キャッシュをクリアして、run に関連付けられている最新の値を取得する必要があります。

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L638-L692)

```python
from_path(
    path
)
```

パスから run、sweep、project、または report を返します。

#### 例:

```
project = api.from_path("my_project")
team_project = api.from_path("my_team/my_project")
run = api.from_path("my_team/my_project/runs/id")
sweep = api.from_path("my_team/my_project/sweeps/id")
report = api.from_path("my_team/my_project/reports/My-Report-Vm11dsdf")
```

| Args |  |
| :--- | :--- |
|  `path` |  (str) project、run、sweep、または report へのパス |

| Returns |  |
| :--- | :--- |
|  `Project`、`Run`、`Sweep`、または `BetaReport` インスタンス。 |

| Raises |  |
| :--- | :--- |
|  パスが無効であるか、オブジェクトが存在しない場合は wandb.Error |

### `job`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1323-L1340)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

指定されたパラメータから `Job` を返します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) ジョブ名。 |
|  `path` |  (str, オプション) 指定された場合、ジョブ artifact をダウンロードするルート パス。 |

| Returns |  |
| :--- | :--- |
|  `Job` オブジェクト。 |

### `list_jobs`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1342-L1418)

```python
list_jobs(
    entity: str,
    project: str
) -> List[Dict[str, Any]]
```

指定された entity および project のジョブ (存在する場合) のリストを返します。

| Args |  |
| :--- | :--- |
|  `entity` |  (str) リストされたジョブの entity。 |
|  `project` |  (str) リストされたジョブの project。 |

| Returns |  |
| :--- | :--- |
|  一致するジョブのリスト。 |

### `project`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L785-L808)

```python
project(
    name: str,
    entity: Optional[str] = None
) -> "public.Project"
```

指定された名前 (および指定された場合は entity) を持つ `Project` を返します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) project 名。 |
|  `entity` |  (str) 要求された entity の名前。None の場合、`Api` に渡されたデフォルトの entity にフォールバックします。デフォルトの entity がない場合は、`ValueError` が発生します。 |

| Returns |  |
| :--- | :--- |
|  `Project` オブジェクト。 |

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L759-L783)

```python
projects(
    entity: Optional[str] = None,
    per_page: Optional[int] = 200
) -> "public.Projects"
```

指定された entity の projects を取得します。

| Args |  |
| :--- | :--- |
|  `entity` |  (str) 要求された entity の名前。None の場合、`Api` に渡されたデフォルトの entity にフォールバックします。デフォルトの entity がない場合は、`ValueError` が発生します。 |
|  `per_page` |  (int) クエリ ページネーションのページ サイズを設定します。None を指定すると、デフォルト サイズが使用されます。通常、これを変更する理由はありません。 |

| Returns |  |
| :--- | :--- |
|  `Project` オブジェクトの反復可能な collection である `Projects` オブジェクト。 |

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1054-L1075)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

パスに基づいて、単一の queue に入れられた run を返します。

entity/project/queue_id/run_queue_item_id の形式のパスを解析します。

### `registries`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1461-L1524)

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) -> Registries
```

Registry イテレーターを返します。

イテレーターを使用して、組織の registry 全体で registry、collection、または artifact バージョンを検索およびフィルタリングします。

#### 例:

名前に "model" が含まれるすべての registry を検索します

```python
import wandb

api = wandb.Api()  # entity が複数の org に属している場合は org を指定します
api.registries(filter={"name": {"$regex": "model"}})
```

名前が "my_collection" でタグが "my_tag" の registry 内のすべての collection を検索します

```python
api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
```

名前が "my_collection" を含み、エイリアスが "best" のバージョンを持つ registry 内のすべての artifact バージョンを検索します

```python
api.registries().collections(
    filter={"name": {"$regex": "my_collection"}}
).versions(filter={"alias": "best"})
```

"model" を含み、タグ "prod" またはエイリアス "best" を持つ registry 内のすべての artifact バージョンを検索します

```python
api.registries(filter={"name": {"$regex": "model"}}).versions(
    filter={"$or": [{"tag": "prod"}, {"alias": "best"}]}
)
```

| Args |  |
| :--- | :--- |
|  `organization` |  (str, オプション) フェッチする registry の組織。指定されていない場合は、ユーザーの設定で指定された組織を使用します。 |
|  `filter` |  (dict, オプション) registry イテレーターの各オブジェクトに適用する MongoDB スタイルのフィルター。collection でフィルタリングに使用できるフィールドは、`name`、`description`、`created_at`、`updated_at` です。collection でフィルタリングに使用できるフィールドは、`name`、`tag`、`description`、`created_at`、`updated_at` です。バージョンでフィルタリングに使用できるフィールドは、`tag`、`alias`、`created_at`、`updated_at`、`metadata` です |

| Returns |  |
| :--- | :--- |
|  registry イテレーター。 |

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L810-L841)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) -> "public.Reports"
```

指定された project パスの reports を取得します。

警告: この API はベータ版であり、将来のリリースで変更される可能性があります

| Args |  |
| :--- | :--- |
|  `path` |  (str) report が存在する project へのパス。形式は "entity/project" である必要があります |
|  `name` |  (str, オプション) 要求された report のオプションの名前。 |
|  `per_page` |  (int) クエリ ページネーションのページ サイズを設定します。None を指定すると、デフォルト サイズが使用されます。通常、これを変更する理由はありません。 |

| Returns |  |
| :--- | :--- |
|  `BetaReport` オブジェクトの反復可能な collection である `Reports` オブジェクト。 |

### `run`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1037-L1052)

```python
run(
    path=""
)
```

entity/project/run_id の形式でパスを解析して、単一の run を返します。

| Args |  |
| :--- | :--- |
|  `path` |  (str) `entity/project/run_id` の形式の run へのパス。`api.entity` が設定されている場合は、`project/run_id` の形式にすることができ、`api.project` が設定されている場合は、run_id のみにすることができます。 |

| Returns |  |
| :--- | :--- |
|  `Run` オブジェクト。 |

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1077-L1090)

```python
run_queue(
    entity, name
)
```

entity の名前付き `RunQueue` を返します。

新しい `RunQueue` を作成するには、`wandb.Api().create_run_queue(...)` を使用します。

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L904-L1035)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

指定されたフィルターに一致する project から run のセットを返します。

フィルターできるフィールドには、次のものがあります。

- `createdAt`: run が作成されたタイムスタンプ。(ISO 8601 形式、例: "2023-01-01T12:00:00Z")
- `displayName`: run の人間が判読できる表示名。(例: "eager-fox-1")
- `duration`: run の合計実行時間 (秒単位)。
- `group`: 関連する run をまとめて整理するために使用されるグループ名。
- `host`: run が実行されたホスト名。
- `jobType`: run のジョブの種類または目的。
- `name`: run の一意の識別子。(例: "a1b2cdef")
- `state`: run の現在の状態。
- `tags`: run に関連付けられているタグ。
- `username`: run を開始したユーザーのユーザー名

さらに、run config または summary metrics の項目でフィルタリングできます。
`config.experiment_name`、`summary_metrics.loss` など。

より複雑なフィルタリングを行うには、MongoDB クエリ演算子を使用できます。
詳細については、https://docs.mongodb.com/manual/reference/operator/query を参照してください。
次の操作がサポートされています。

- `$and`
- `$or`
- `$nor`
- `$eq`
- `$ne`
- `$gt`
- `$gte`
- `$lt`
- `$lte`
- `$in`
- `$nin`
- `$exists`
- `$regex`

#### 例:

config.experiment_name が "foo" に設定されている my_project で run を検索します

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": "foo"},
)
```

config.experiment_name が "foo" または "bar" に設定されている my_project で run を検索します

```
api.runs(
    path="my_entity/my_project",
    filters={
        "$or": [
            {"config.experiment_name": "foo"},
            {"config.experiment_name": "bar"},
        ]
    },
)
```

config.experiment_name が正規表現に一致する my_project で run を検索します (アンカーはサポートされていません)

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
```

run 名が正規表現に一致する my_project で run を検索します (アンカーはサポートされていません)

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}},
)
```

config.experiment に値 "testing" を持つネストされたフィールド "category" が含まれている my_project で run を検索します

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment.category": "testing"},
)
```

summary metrics の model1 の下の辞書にネストされた損失値が 0.5 の my_project で run を検索します

```
api.runs(
    path="my_entity/my_project",
    filters={"summary_metrics.model1.loss": 0.5},
)
```

損失の昇順でソートされた my_project で run を検索します

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| Args |  |
| :--- | :--- |
|  `path` |  (str) project へのパス。形式は "entity/project" である必要があります |
|  `filters` |  (dict) MongoDB クエリ言語を使用して特定の run をクエリします。config.key、summary_metrics.key、state、entity、createdAt などの run プロパティでフィルタリングできます。たとえば、`{"config.experiment_name": "foo"}` は、experiment 名が "foo" に設定された config エントリを持つ run を検索します |
|  `order` |  (str) 順序は、`created_at`、`heartbeat_at`、`config.*.value`、または `summary_metrics.*` にすることができます。順序の前に + を付けると、順序は昇順になります。順序の前に - を付けると、順序は降順になります (デフォルト)。デフォルトの順序は、run.created_at が最も古いものから最も新しいものになります。 |
|  `per_page` |  (int) クエリ ページネーションのページ サイズを設定します。 |
|  `include_sweeps` |  (bool) 結果に sweep runs を含めるかどうか。 |

| Returns |  |
| :--- | :--- |
|  `Run` オブジェクトの反復可能な collection である `Runs` オブジェクト。 |

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1092-L1107)

```python
sweep(
    path=""
)
```

`entity/project/sweep_id` の形式でパスを解析して sweep を返します。

| Args |  |
| :--- | :--- |
|  `path` |  (str, optional) entity/project/sweep_id の形式の sweep へのパス。`api.entity` が設定されている場合は、project/sweep_id の形式にすることができ、`api.project` が設定されている場合は、sweep_id のみにすることができます。 |

| Returns |  |
| :--- | :--- |
|  `Sweep` オブジェクト。 |

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L564-L586)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

tfevent ファイルを含むローカル ディレクトリを wandb に同期します。

### `team`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L855-L864)

```python
team(
    team: str
) -> "public.Team"
```

指定された名前を持つ一致する `Team` を返します。

| Args |  |
| :--- | :--- |
|  `team` |  (str) team の名前。 |

| Returns |  |
| :--- | :--- |
|  `Team` オブジェクト。 |

### `upsert_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L437-L550)

```python
upsert_run_queue(
    name: str,
    resource_config: dict,
    resource_type: "public.RunQueueResourceType",
    entity: Optional[str] = None,
    template_variables: Optional[dict] = None,
    external_links: Optional[dict] = None,
    prioritization_mode: Optional['public.RunQueuePrioritizationMode'] = None
)
```

run queue (Launch) をアップサートします。

| Args |  |
| :--- | :--- |
|  `name` |  (str) 作成する queue の名前 |
|  `entity` |  (str) queue を作成する entity のオプションの名前。None の場合、設定されたまたはデフォルトの entity が使用されます。 |
|  `resource_config` |  (dict) queue に使用するオプションのデフォルト リソース設定。handlebars (例: `{{var}}`) を使用してテンプレート変数を指定します。 |
|  `resource_type` |  (str) queue に使用するリソースの型。"local-container"、"local-process"、"kubernetes"、"sagemaker"、または "gcp-vertex" のいずれか。 |
|  `template_variables` |  (dict) config で使用するテンプレート変数スキーマの辞書。予期される形式: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } }` |
|  `external_links` |  (dict) queue で使用する外部リンクのオプションの辞書。予期される形式: `{ "name": "url" }` |
|  `prioritization_mode` |  (str) 使用する優先順位付けのオプションのバージョン。"V0" または None |

| Returns |  |
| :--- | :--- |
|  アップサートされた `RunQueue`。 |

| Raises |  |
| :--- | :--- |
|  パラメータが無効な場合は ValueError wandb API エラーの場合は wandb.Error |

### `user`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L866-L886)

```python
user(
    username_or_email: str
) -> Optional['public.User']
```

ユーザー名またはメール アドレスからユーザーを返します。

注: この関数はローカル管理者に対してのみ機能します。自分のユーザー オブジェクトを取得しようとしている場合は、`api.viewer` を使用してください。

| Args |  |
| :--- | :--- |
|  `username_or_email` |  (str) ユーザーのユーザー名またはメール アドレス |

| Returns |  |
| :--- | :--- |
|  `User` オブジェクト。ユーザーが見つからない場合は None |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L888-L902)

```python
users(
    username_or_email: str
) -> List['public.User']
```

部分的なユーザー名またはメール アドレス クエリからすべてのユーザーを返します。

注: この関数はローカル管理者に対してのみ機能します。自分のユーザー オブジェクトを取得しようとしている場合は、`api.viewer` を使用してください。

| Args |  |
| :--- | :--- |
|  `username_or_email` |  (str) 検索するユーザーのプレフィックスまたはサフィックス |

| Returns |  |
| :--- | :--- |
|  `User` オブジェクトの配列 |

| Class Variables |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `DEFAULT_ENTITY_QUERY`<a id="DEFAULT_ENTITY_QUERY"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |
