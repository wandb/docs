---
title: 'Api


  API (Application Programming Interface) は、プログラム間でデータや機能をやり取りするためのインターフェースを提供します。W&B
  API は、ユーザーがwandb を通じて**Projects**、**Entities**、**Runs**などのデータにアクセスし、管理することを可能にします。このAPIにより、**Experiments**の追跡や**Reports**の共有を自動化できます。W&B
  APIはPythonを含むさまざまなプログラミング言語で利用可能です。これにより、カスタムスクリプトの作成や他のツールとの統合が簡単になります。APIの使用方法に関する詳細な情報は、[API
  documentation](https://wandb.ai)を参照してください。'
menu:
  reference:
    identifier: ja-ref-python-public-api-api
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L104-L1573 >}}

wandb サーバーをクエリするために使用されます。

```python
Api(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) -> None
```

#### 例:

最も一般的な初期化方法

```
>>> wandb.Api()
```

| 引数 |  |
| :--- | :--- |
|  `overrides` |  (dict) `https://api.wandb.ai` 以外の wandb サーバーを使用している場合に `base_url` を設定できます。また、`entity`、`project`、および `run` のデフォルト設定をすることができます。 |

| 属性 |  |
| :--- | :--- |

## メソッド

### `artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1299-L1321)

```python
artifact(
    name: str,
    type: Optional[str] = None
)
```

`project/name` または `entity/project/name` の形式でパスを解析することにより、単一のアーティファクトを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) アーティファクト名。project/ または entity/project/ で始まる場合があります。name に entity が指定されていない場合、Run または API 設定の entity が使用されます。有効な名前は次の形式になります: name:version name:alias |
|  `type` |  (str, オプション) 取得するアーティファクトのタイプ。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ValueError` |  アーティファクト名が指定されていない場合。 |
|  `ValueError` |  アーティファクトタイプが指定されているが、取得したアーティファクトのタイプと一致しない場合。 |

#### 注意:

このメソッドは外部利用のみを目的としています。wandb リポジトリコード内で `api.artifact()` を呼び出さないでください。

### `artifact_collection`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1181-L1210)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

タイプと `entity/project/name` の形式でパスを解析することにより、単一のアーティファクトコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するアーティファクトコレクションのタイプ。 |
|  `name` |  (str) アーティファクトコレクション名。entity/project で始まる場合があります。 |

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactCollection` オブジェクト。 |

### `artifact_collection_exists`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1442-L1459)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

指定されたプロジェクトとエンティティ内にアーティファクトコレクションが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) アーティファクトコレクション名。entity/project で始まる場合があります。entity または project が指定されていない場合、オーバーライドパラメーターから推測されます。その他の場合、entity はユーザー設定から取得され、project は "uncategorized" にデフォルト設定されます。 |
|  `type` |  (str) アーティファクトコレクションのタイプ |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトコレクションが存在する場合は True、そうでない場合は False。 |

### `artifact_collections`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1154-L1179)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) -> "public.ArtifactCollections"
```

一致するアーティファクトコレクションのコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project_name` |  (str) フィルタリングするプロジェクトの名前。 |
|  `type_name` |  (str) フィルタリングするアーティファクトタイプの名前。 |
|  `per_page` |  (int, オプション) クエリのページネーションのページサイズを設定します。None はデフォルトサイズを使用します。通常、これを変更する理由はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  イテラブルな `ArtifactCollections` オブジェクト。 |

### `artifact_exists`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1420-L1440)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

指定されたプロジェクトとエンティティ内にアーティファクトバージョンが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) アーティファクト名。entity/project で始まる場合があります。entity または project が指定されていない場合、オーバーライドパラメータから推測されます。その他の場合、entity はユーザー設定から取得され、project は "uncategorized" にデフォルト設定されます。有効な名前は次の形式になります: name:version name:alias |
|  `type` |  (str, オプション) アーティファクトのタイプ |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトバージョンが存在する場合は True、そうでない場合は False。 |

### `artifact_type`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1130-L1152)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

一致する `ArtifactType` を返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するアーティファクトタイプの名前。 |
|  `project` |  (str, オプション) 指定されている場合、フィルタリングするプロジェクト名またはパス。 |

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactType` オブジェクト。 |

### `artifact_types`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1109-L1128)

```python
artifact_types(
    project: Optional[str] = None
) -> "public.ArtifactTypes"
```

一致するアーティファクトタイプのコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project` |  (str, オプション) 指定されている場合、フィルタリングするプロジェクト名またはパス。 |

| 戻り値 |  |
| :--- | :--- |
|  イテラブルな `ArtifactTypes` オブジェクト。 |

### `artifact_versions`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1212-L1222)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

非推奨、代わりに `artifacts(type_name, name)` を使用してください。

### `artifacts`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1224-L1260)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: Optional[int] = 50,
    tags: Optional[List[str]] = None
) -> "public.Artifacts"
```

指定されたパラメータから `Artifacts` コレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するアーティファクトのタイプ。 |
|  `name` |  (str) アーティファクトコレクションの名前。entity/project で始まる場合があります。 |
|  `per_page` |  (int, オプション) クエリのページネーションのページサイズを設定します。None はデフォルトサイズを使用します。通常、これを変更する理由はありません。 |
|  `tags` |  (list[str], オプション) これらのタグがすべて含まれているアーティファクトのみを返します。 |

| 戻り値 |  |
| :--- | :--- |
|  イテラブルな `Artifacts` オブジェクト。 |

### `create_project`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L294-L301)

```python
create_project(
    name: str,
    entity: str
) -> None
```

新しいプロジェクトを作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 新しいプロジェクトの名前。 |
|  `entity` |  (str) 新しいプロジェクトのエンティティ。 |

### `create_run`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L303-L323)

```python
create_run(
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) -> "public.Run"
```

新しい run を作成します。

| 引数 |  |
| :--- | :--- |
|  `run_id` |  (str, オプション) 指定された場合、run に割り当てられる ID。run ID はデフォルトで自動生成されますので、通常はこれを指定する必要はありません。指定する場合はリスクを負ってください。 |
|  `project` |  (str, オプション) 指定された場合、新しい run のプロジェクト。 |
|  `entity` |  (str, オプション) 指定された場合、新しい run のエンティティ。 |

| 戻り値 |  |
| :--- | :--- |
|  新たに作成された `Run`。 |

### `create_run_queue`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L325-L435)

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

新しい run キュー (launch) を作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 作成するキューの名前 |
|  `type` |  (str) キューに使用されるリソースのタイプ。"local-container"、"local-process"、"kubernetes"、"sagemaker"、または "gcp-vertex" のいずれか。 |
|  `entity` |  (str) キューを作成するエンティティのオプションの名前。None の場合、設定されたまたはデフォルトのエンティティが使用されます。 |
|  `prioritization_mode` |  (str) オプションのプライオリティバージョン。"V0" または None |
|  `config` |  (dict) キューに使用されるデフォルトのリソース設定のオプション。テンプレート変数を指定するにはハンドルバー（例：`{{var}}`）を使用します。 |
|  `template_variables` |  (dict) 設定内で使用されるテンプレート変数のスキーマの辞書。期待される形式: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } }` |

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `RunQueue` |

| 例外 |  |
| :--- | :--- |
|  ValueError: パラメーターのいずれかが無効な場合 wandb.Error: wandb API のエラー |

### `create_team`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L843-L853)

```python
create_team(
    team, admin_username=None
)
```

新しいチームを作成します。

| 引数 |  |
| :--- | :--- |
|  `team` |  (str) チーム名 |
|  `admin_username` |  (str) チームの管理ユーザーのオプションのユーザー名、デフォルトは現在のユーザーです。 |

| 戻り値 |  |
| :--- | :--- |
|  `Team` オブジェクト |

### `create_user`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L552-L562)

```python
create_user(
    email, admin=(False)
)
```

新しいユーザーを作成します。

| 引数 |  |
| :--- | :--- |
|  `email` |  (str) ユーザーのメールアドレス |
|  `admin` |  (bool) このユーザーがグローバルインスタンス管理者であるかどうか |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクト |

### `flush`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L629-L636)

```python
flush()
```

ローカルキャッシュをフラッシュします。

API オブジェクトは run のローカルキャッシュを保持するため、スクリプトを実行中に run の状態が変更される可能性がある場合、`api.flush()` を使用してローカルキャッシュをクリアし、run に関連付けられた最新の値を取得します。

### `from_path`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L638-L692)

```python
from_path(
    path
)
```

パスから run、sweep、プロジェクト、またはレポートを返します。

#### 例:

```
project = api.from_path("my_project")
team_project = api.from_path("my_team/my_project")
run = api.from_path("my_team/my_project/runs/id")
sweep = api.from_path("my_team/my_project/sweeps/id")
report = api.from_path("my_team/my_project/reports/My-Report-Vm11dsdf")
```

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) プロジェクト、run、sweep、またはレポートへのパス |

| 戻り値 |  |
| :--- | :--- |
|  `Project`、`Run`、`Sweep`、または `BetaReport` インスタンス。 |

| 例外 |  |
| :--- | :--- |
|  wandb.Error: パスが無効、またはオブジェクトが存在しない場合 |

### `job`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1323-L1340)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

指定されたパラメーターから `Job` を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) ジョブの名前。 |
|  `path` |  (str, オプション) 指定された場合、ジョブアーティファクトをダウンロードするルートパス。 |

| 戻り値 |  |
| :--- | :--- |
|  `Job` オブジェクト。 |

### `list_jobs`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1342-L1418)

```python
list_jobs(
    entity: str,
    project: str
) -> List[Dict[str, Any]]
```

指定されたエンティティとプロジェクトに対して、利用可能なジョブのリストを返します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str) リストされたジョブのエンティティ。 |
|  `project` |  (str) リストされたジョブのプロジェクト。 |

| 戻り値 |  |
| :--- | :--- |
|  一致するジョブのリスト。 |

### `project`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L785-L808)

```python
project(
    name: str,
    entity: Optional[str] = None
) -> "public.Project"
```

指定された名前 (および指定された場合はエンティティ) の `Project` を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) プロジェクト名。 |
|  `entity` |  (str) リクエストされたエンティティ名。None の場合、`Api` に渡されたデフォルトのエンティティにフォールバックします。デフォルトのエンティティがない場合は、`ValueError` をスローします。 |

| 戻り値 |  |
| :--- | :--- |
|  `Project` オブジェクト。 |

### `projects`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L759-L783)

```python
projects(
    entity: Optional[str] = None,
    per_page: Optional[int] = 200
) -> "public.Projects"
```

指定されたエンティティのプロジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str) リクエストされたエンティティ名。None の場合、`Api` に渡されたデフォルトのエンティティにフォールバックします。デフォルトのエンティティがない場合は、`ValueError` をスローします。 |
|  `per_page` |  (int) クエリのページネーションのページサイズを設定します。None はデフォルトサイズを使用します。通常、これを変更する理由はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  `Projects` オブジェクトで、`Project` オブジェクトのイテラブルなコレクションです。 |

### `queued_run`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1054-L1075)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

パスに基づいて単一のキューされた run を返します。

`entity/project/queue_id/run_queue_item_id` の形式のパスを解析します。

### `registries`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1461-L1524)

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) -> Registries
```

レジストリのイテレータを返します。

イテレータを使用して、組織のレジストリ内のレジストリ、コレクション、またはアーティファクトバージョンを検索およびフィルタリングします。

#### 例:

"model" を含む名前のすべてのレジストリを見つけます。

```python
import wandb

api = wandb.Api()  # エンティティが複数の組織に属する場合、組織を指定します。
api.registries(filter={"name": {"$regex": "model"}})
```

"my_collection" という名前と "my_tag" というタグのあるコレクションをレジストリで見つけます。

```python
api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
```

"my_collection" を含むコレクション名と "best" というエイリアスを持つバージョンのあるすべてのアーティファクトバージョンを見つけます。

```python
api.registries().collections(
    filter={"name": {"$regex": "my_collection"}}
).versions(filter={"alias": "best"})
```

"model" を含み、タグ "prod" またはエイリアス "best" を持つすべてのアーティファクトバージョンをレジストリで見つけます。

```python
api.registries(filter={"name": {"$regex": "model"}}).versions(
    filter={"$or": [{"tag": "prod"}, {"alias": "best"}]}
)
```

| 引数 |  |
| :--- | :--- |
|  `organization` |  (str, オプション) 取得するレジストリの組織。指定されていない場合、ユーザー設定で指定された組織を使用します。 |
|  `filter` |  (dict, オプション) レジストリイテレータ内の各オブジェクトに適用する MongoDB スタイルのフィルタ。コレクションをフィルタリングする際に利用可能なフィールド: `name`, `description`, `created_at`, `updated_at`。コレクションをフィルタリングする際に利用可能なフィールド: `name`, `tag`, `description`, `created_at`, `updated_at`。バージョンをフィルタリングする際に利用可能なフィールド: `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

| 戻り値 |  |
| :--- | :--- |
|  レジストリのイテレータ。 |

### `reports`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L810-L841)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) -> "public.Reports"
```

指定されたプロジェクトパスのレポートを取得します。

警告: この API はベータ版であり、将来のリリースで変更される可能性があります。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) レポートが存在するプロジェクトのパス、形式は: "entity/project" となります。 |
|  `name` |  (str, オプション) リクエストされたレポートのオプションの名前。 |
|  `per_page` |  (int) クエリのページネーションのページサイズを設定します。None はデフォルトサイズを使用します。通常、これを変更する理由はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  `Reports` オブジェクトで、`BetaReport` オブジェクトのイテラブルなコレクションです。 |

### `run`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1037-L1052)

```python
run(
    path=""
)
```

`entity/project/run_id` の形式でパスを解析することにより、単一の run を返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) `entity/project/run_id` 形式の run へのパス。`api.entity` が設定されている場合、この形式は `project/run_id` となり、`api.project` が設定されている場合、run_id のみです。 |

| 戻り値 |  |
| :--- | :--- |
|  `Run` オブジェクト。 |

### `run_queue`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1077-L1090)

```python
run_queue(
    entity, name
)
```

エンティティの名前付き `RunQueue` を返します。

新しい `RunQueue` を作成するには、`wandb.Api().create_run_queue(...)` を使用してください。

### `runs`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L904-L1035)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

指定されたフィルターに一致するプロジェクトからの一連の run を返します。

フィルターに使用できるフィールドには以下のものがあります:

- `createdAt`: run が作成されたタイムスタンプ。（ISO 8601 フォーマット、例: "2023-01-01T12:00:00Z"）
- `displayName`: run の人間が読みやすい表示名。（例: "eager-fox-1"）
- `duration`: run の合計実行時間（秒単位）。
- `group`: 関連する run をまとめるために使用されるグループ名。
- `host`: run が実行されたホスト名。
- `jobType`: ジョブのタイプまたは run の目的。
- `name`: run の一意の識別子。（例: "a1b2cdef"）
- `state`: run の現在の状態。
- `tags`: run に関連付けられたタグ。
- `username`: run を開始したユーザーのユーザー名。

さらに、run の設定や要約メトリクス内の項目によるフィルタリングが可能です。例: `config.experiment_name`, `summary_metrics.loss` など。

より複雑なフィルタリングには、MongoDB クエリオペレーターを使用できます。詳細は、以下を参照してください: https://docs.mongodb.com/manual/reference/operator/query サポートされている操作には以下のものがあります:

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

設定されている foo という experiment_name を持つ my_project 内の run を見つけます

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": "foo"},
)
```

設定されている foo または bar という experiment_name を持つ my_project 内の run を見つけます

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

experiment_name が正規表現に一致する my_project 内の run を見つけます（アンカーはサポートされていません）

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
```

run の名前が正規表現に一致する my_project 内の run を見つけます（アンカーはサポートされていません）

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}},
)
```

実験に "category" というネストされたフィールドを持つ run を探します

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment.category": "testing"},
)
```

要約メトリクスの model1 下に辞書としてネストされている損失値0.5を持つ run を探します

```
api.runs(
    path="my_entity/my_project",
    filters={"summary_metrics.model1.loss": 0.5},
)
```

上昇損失に基づいて my_project 内の run を探します

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) プロジェクトへのパス。形式は: "entity/project" |
|  `filters` |  (dict) MongoDB クエリ言語を使用して特定の run をクエリします。run のプロパティ（config.key、summary_metrics.key、state、entity、createdAt など）でフィルタリングできます。例: `{"config.experiment_name": "foo"}` は、実験名に foo が設定されている run を見つけます。 |
|  `order` |  (str) 並び順は `created_at`、`heartbeat_at`、`config.*.value`、`summary_metrics.*` にできます。order の前に + を付けると昇順になります。order の前に - を付けると降順（デフォルト）になります。デフォルトの並び順は、run の created_at で、古い順から新しい順です。 |
|  `per_page` |  (int) クエリのページネーションのページサイズを設定します。 |
|  `include_sweeps` |  (bool) 結果に sweep run を含めるかどうか。 |

| 戻り値 |  |
| :--- | :--- |
|  `Runs` オブジェクトで、`Run` オブジェクトのイテラブルなコレクションです。 |

### `sweep`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L1092-L1107)

```python
sweep(
    path=""
)
```

`sweep_id` の形式でパスを解析することにより、sweep を返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str, オプション) エンティティまたはプロジェクトの設定がされていない場合、sweep に対するパスの形式は entity/project/sweep_id である必要があります。`api.entity` が設定されている場合、この形式は project/sweep_id になり、`api.project` が設定されている場合、sweep_id のみです。 |

| 戻り値 |  |
| :--- | :--- |
|  `Sweep` オブジェクト。 |

### `sync_tensorboard`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L564-L586)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

tfevent ファイルを含むローカルディレクトリを wandb に同期します。

### `team`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L855-L864)

```python
team(
    team: str
) -> "public.Team"
```

指定された名前の `Team` を返します。

| 引数 |  |
| :--- | :--- |
|  `team` |  (str) チーム名。 |

| 戻り値 |  |
| :--- | :--- |
|  `Team` オブジェクト。 |

### `upsert_run_queue`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L437-L550)

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

run キュー (launch) をアップサートします。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 作成するキューの名前 |
|  `entity` |  (str) 作成するキューのエンティティのオプションの名前。None の場合、設定されたまたはデフォルトのエンティティを使用します。 |
|  `resource_config` |  (dict) キューに使用されるデフォルトのリソース設定のオプション。テンプレート変数を指定するにはハンドルバー（例：`{{var}}`）を使用します。 |
|  `resource_type` |  (str) キューに使用されるリソースのタイプ。"local-container"、"local-process"、"kubernetes"、"sagemaker"、または "gcp-vertex" のいずれか。 |
|  `template_variables` |  (dict) 設定内で使用されるテンプレート変数のスキーマの辞書。期待される形式: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } }` |
|  `external_links` |  (dict) キューで使用される外部リンクのオプションの辞書。期待される形式: `{ "name": "url" }` |
|  `prioritization_mode` |  (str) 使用するプライオリティのバージョン。 "V0" または None |

| 戻り値 |  |
| :--- | :--- |
|  アップサートされた `RunQueue`。 |

| 例外 |  |
| :--- | :--- |
|  ValueError: パラメーターのいずれかが無効な場合 wandb.Error: wandb API のエラー |

### `user`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L866-L886)

```python
user(
    username_or_email: str
) -> Optional['public.User']
```

ユーザー名またはメールアドレスからユーザーを返します。

注意: この関数はローカル管理者のみ機能します。 自分のユーザーオブジェクトを取得しようとしている場合は `api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) ユーザーのユーザー名またはメールアドレス |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクトまたはユーザーが見つからない場合は None |

### `users`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/api.py#L888-L902)

```python
users(
    username_or_email: str
) -> List['public.User']
```

部分的なユーザー名またはメールアドレスクエリからすべてのユーザーを返します。

注意: この関数はローカル管理者のみ機能します。 自分のユーザーオブジェクトを取得しようとしている場合は `api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) 検索したいユーザーのプレフィックスまたはサフィックス |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクトの配列 |

| クラス変数 |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `DEFAULT_ENTITY_QUERY`<a id="DEFAULT_ENTITY_QUERY"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |