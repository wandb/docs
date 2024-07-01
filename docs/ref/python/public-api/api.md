# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L95-L1179' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandbサーバーをクエリするために使用します。

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
|  `overrides` |  (辞書) 別のwandbサーバーを使用する場合は、`base_url` を設定することができます。また、`entity`、`project`、および `run` のデフォルト値を設定することもできます。 |

| 属性 |  |
| :--- | :--- |

## メソッド

### `artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1017-L1041)

```python
artifact(
    name, type=None
)
```

`entity/project/name` の形式でパスを解析して単一のartifactを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) アーティファクトの名前。`entity/project`で始まる可能性があります。 有効な名前の形式は次のとおりです: name: version name: alias |
|  `type` |  (str, オプション) 取得するアーティファクトのタイプ。|

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L967-L983)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

タイプと次の形式でパスを解析して単一のartifactコレクションを返します: `entity/project/name`

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するアーティファクトコレクションのタイプ。 |
|  `name` |  (str) アーティファクトコレクションの名前。`entity/project`で始まる可能性があります。|

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactCollection`オブジェクト。 |

### `artifact_collection_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1162-L1179)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

指定されたprojectとentity内にアーティファクトコレクションが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) アーティファクトコレクション名。`entity/project`で始まる可能性があります。エンティティまたはプロジェクトが指定されていない場合、オーバーライドパラメータから推測されます。別の設定がない場合、エンティティはユーザーの設定から取得され、プロジェクトは「未分類」にデフォルトされます。 |
|  `type` |  (str) アーティファクトコレクションのタイプ |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトコレクションが存在する場合はTrue、そうでない場合はFalse。 |

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L947-L965)

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
|  `project_name` |  (str) 絞り込みを行うプロジェクトの名前。|
|  `type_name` |  (str) 絞り込みを行うアーティファクトのタイプの名前。|
|  `per_page` |  (int, オプション) クエリページネーションのページサイズを設定します。Noneはデフォルトサイズを使用します。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  繰り返し可能な `ArtifactCollections`オブジェクト。|

### `artifact_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1140-L1160)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

指定されたprojectとentity内にアーティファクトのバージョンが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) アーティファクトの名前。`entity/project`で始まる可能性があります。 エンティティまたはプロジェクトが指定されていない場合、オーバーライドパラメータから推測されます。別の設定がない場合、エンティティはユーザーの設定から取得され、プロジェクトは「未分類」にデフォルトされます。有効な名前の形式は次のとおりです: name: version name: alias |
|  `type` |  (str, オプション) アーティファクトのタイプ |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトのバージョンが存在する場合はTrue、そうでない場合はFalse。|

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L931-L945)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

一致する`ArtifactType`を返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するアーティファクトのタイプの名前。|
|  `project` |  (str, オプション) 指定する場合、プロジェクト名またはフィルターに使用するパス。|

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactType` オブジェクト。|

### `artifact_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L918-L929)

```python
artifact_types(
    project: Optional[str] = None
) -> "public.ArtifactTypes"
```

一致するアーティファクトタイプのコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project` |  (str, オプション) 指定する場合、プロジェクト名またはフィルターに使用するパス。|

| 戻り値 |  |
| :--- | :--- |
|  繰り返し可能な `ArtifactTypes` オブジェクト。|

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L985-L995)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

非推奨、代わりに `artifacts(type_name, name)` を使用してください。

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L997-L1015)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: Optional[int] = 50
) -> "public.Artifacts"
```

指定されたパラメータから `Artifacts` コレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するアーティファクトのタイプ。|
|  `name` |  (str) アーティファクトコレクションの名前。`entity/project`で始まる可能性があります。|
|  `per_page` |  (int, オプション) クエリページネーションのページサイズを設定します。Noneはデフォルトサイズを使用します。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  繰り返し可能な `Artifacts` オブジェクト。|

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L281-L288)

```python
create_project(
    name: str,
    entity: str
) -> None
```

新しいプロジェクトを作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 新しいプロジェクトの名前。|
|  `entity` |  (str) 新しいプロジェクトのエンティティ。|

### `create_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L290-L310)

```python
create_run(
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) -> "public.Run"
```

新しいRunを作成します。

| 引数 |  |
| :--- | :--- |
|  `run_id` |  (str, オプション) 付与するRunのID。指定しない場合は自動生成されるので、通常は指定する必要はありません。 |
|  `project` |  (str, オプション) 指定する場合、新しいRunのプロジェクト。|
|  `entity` |  (str, オプション) 指定する場合、新しいRunのエンティティ。|

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `Run` 。|

### `create_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L312-L422)

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

新しいRunキュー（ローンンチ）を作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 作成するキューの名前 |
|  `type` |  (str) キューに使用するリソースのタイプ。次のいずれか: "local-container", "local-process", "kubernetes", "sagemaker", または "gcp-vertex"。|
|  `entity` |  (str) オプションのエンティティ名。Noneの場合、構成済みまたはデフォルトのエンティティが使用されます。|
|  `prioritization_mode` |  (str) オプションの優先順位モード。 "V0" または None|
|  `config` |  (辞書) オプションのデフォルトリソース設定。テンプレート変数を指定するためにハンドルバー（例: "{{var}}"）を使用します。|
|  `template_variables` |  (辞書) 設定に使用するテンプレート変数スキーマの辞書。期待される形式は次のとおり: { "var-name": { "schema": { "type": ("string", "number", または "integer"), "default": (オプションの値), "minimum": (オプションの最小値), "maximum": (オプションの最大値), "enum": [..."(オプション)"] } } } |

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `RunQueue` |

| 例外 |  |
| :--- | :--- |
|  値が無効な場合はValueError, wandb APIエラー時はwandb.Error |

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L705-L715)

```python
create_team(
    team, admin_username=None
)
```

新しいTeamを作成します。

| 引数 |  |
| :--- | :--- |
|  `team` |  (str) チームの名前 |
|  `admin_username` |  (str) オプションのチームの管理ユーザーのユーザー名。デフォルトは現在のユーザーです。|

| 戻り値 |  |
| :--- | :--- |
|  `Team`オブジェクト |

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L424-L434)

```python
create_user(
    email, admin=(False)
)
```

新しいユーザーを作成します。

| 引数 |  |
| :--- | :--- |
|  `email` |  (str) ユーザーのメールアドレス |
|  `admin` |  (bool) このユーザーをグローバルインスタンス管理者にするかどうか |

| 戻り値 |  |
| :--- | :--- |
|  `User`オブジェクト |

### `flush`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L501-L508)

```python
flush()
```

ローカルキャッシュをフラッシュします。

apiオブジェクトはRunのローカルキャッシュを保持しているため、スクリプトを実行している間にRunの状態が変わる可能性がある場合は、`api.flush()` を使用して最新の値を取得する必要があります。

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L510-L564)

```python
from_path(
    path
)
```

指定されたパスからRun、Sweep、ProjectまたはReportを返します。

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
|  `path` |  (str) プロジェクト、Run、SweepまたはReportのパス |

| 戻り値 |  |
| :--- | :--- |
|  `Project`、`Run`、`Sweep`、または `BetaReport` インスタンス。|

| 例外 |  |
| :--- | :--- |
|  パスが無効な場合またはオブジェクトが存在しない場合はwandb.Error |

### `job`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1043-L1060)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

指定されたパラメータから`Job`を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) ジョブの名前。|
|  `path` |  (str, オプション) 指定する場合、ジョブアーティファクトをダウンロードするためのルートパス。|

| 戻り値 |  |
| :--- | :--- |
|  `Job` オブジェクト。|

### `list_jobs`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1062-L1138)

```python
list_jobs(
    entity: str,
    project: str
) -> List[Dict[str, Any]]
```

指定されたエンティティとプロジェクトに対するジョブのリストを返します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str) 列挙されるジョブのエンティティ。|
|  `project` |  (str) 列挙されるジョブのプロジェクト。|

| 戻り値 |  |
| :--- | :--- |
|  一致するジョブのリスト。|

### `project`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L657-L670)

```python
project(
    name: str,
    entity: Optional[str] = None
) -> "public.Project"
```

指定された名前の`Project`を返します（エンティティが与えられた場合はそれも）。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) プロジェクト名。|
|  `entity` |  (str) 要求されたエンティティの名前。Noneの場合、`Api`に渡されたデフォルトのエンティティを使用します。デフォルトのエンティティがない場合は、 `ValueError` を発生させます。|

| 戻り値 |  |
| :--- | :--- |
|  `Project` オブジェクト。|

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L631-L655)

```python
projects(
    entity: Optional[str] = None,
    per_page: Optional[int] = 200
) -> "public.Projects"
```

指定されたエンティティのプロジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str) 要求されたエンティティの名前。Noneの場合、`Api`に渡されたデフォルトのエンティティを使用します。デフォルトのエンティティがない場合は、 `ValueError` を発生させます。|
|  `per_page` |  (int) クエリページネーションのページサイズを設定します。Noneはデフォルトサイズを使用します。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  繰り返し可能な `Projects` オブジェクト。|

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L863-L884)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

パスに基づく単一のキューに入ったRunを返します。

パスの形式: entity/project/queue_id/run_queue_item_id.

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L672-L703)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) -> "public.Reports"
```

指定されたプロジェクトパスのReportsを取得します。

警告: このAPIはベータ版であり、将来のリリースで変更される可能性があります。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) Reportが存在するプロジェクトへのパス。形式は次のとおり: "entity/project" |
|  `name` |  (str, オプション) 要求されたReportの名前。|
|  `per_page` |  (int) クエリページネーションのページサイズを設定します。Noneはデフォルトサイズを使用します。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  繰り返し可能な `Reports` オブジェクト。|

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L846-L861)

```python
run(
    path=""
)
```

entity/project/run_id の形式でパスを解析して単一のRunを返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) `entity/project/run_id` の形式のRunへのパス。 `api.entity` が設定されている場合、 `project/run_id` の形式で、 `api.project` が設定されている場合は、この形式でRun IDを指定できます。|

| 戻り値 |  |
| :--- | :--- |
|  `Run` オブジェクト。|

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L886-L899)

```python
run_queue(
    entity, name
)
```

エンティティの名前付き `RunQueue` を返します。

新しい `RunQueue` を作成するには、 `wandb.Api().create_run_queue(...)` を使用します。

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L766-L844)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

指定されたフィルタに一致するプロジェクトのRunのセットを返します。

`config.*`、 `summary_metrics.*`、 `tags`、 `state`、`entity`、`createdAt` などでフィルタリングできます。

#### 例:

`config.experiment_name` が "foo" に設定されている `my_project` のRunを検索する

```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

`config.experiment_name` が "foo" または "bar" に設定されている `my_project` のRunを検索する

```
api.runs(
    path="my_entity/my_project",
    filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```

`config.experiment_name` が正規表現に一致する `my_project` のRunを検索する（アンカーはサポートされていません）

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

Run名が正規表現に一致する `my_project` のRunを検索する（アンカーはサポートされていません）

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}}
)
```

昇順で損失をソートした `my_project` のRunを検索する

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) プロジェクトへのパス。形式は次のとおり: "entity/project" |
|  `filters` |  (辞書) MongoDBクエリ言語を使用して特定のRunをクエリ。`config.key`、`summary_metrics.key`、`state`、`entity`、`createdAt` などで実行プロパティをフィルタリングできます。 例: {"config.experiment_name": "foo"} は、configエントリの実験名が "foo" に設定されているRunを見つけます。操作を構成してより複雑なクエリを作成できます。言語の参照は https://docs.mongodb.com/manual/reference/operator/query |
|  `order` |  (str) ソート順は `created_at`、`heartbeat_at`、`config.*.value` または `summary_metrics.*` にできます。順番に + を付けると昇順になります。順番に - を付けると降順（デフォルト）になります。 デフォルトでは新しいものから古いものへrun.created_atをソートします。|
|  `per_page` |  (int) クエリページネーションのページサイズを設定します。 |
|  `include_sweeps` |  (bool) ＳweepのRunを結果に含めるかどうか。|

| 戻り値 |  |
| :--- | :--- |
|  繰り返し可能な `Runs` オブジェクト。|

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L901-L916)

```python
sweep(
    path=""
)
```

`entity/project/sweep_id` の形式でパスを解析してSweepを返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str, オプション) `entity/project/sweep_id` の形式のSweepのパス。 `api.entity` が設定されている場合、この形式で `project/sweep_id` として、 `api.project` が設定されている場合、この形式で `sweep_id` として指定できます。|

| 戻り値 |  |
| :--- | :--- |
|  `Sweep` オブジェクト。|

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L436-L458)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

tfeventファイルを含むローカルディレクトリーをwandbと同期します。

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L717-L726)

```python
team(
    team: str
) -> "public.Team"
```

指定された名前の一致する `Team` を返します。

| 引数 |  |
| :--- | :--- |
|  `team` |  (str) チームの名前。|

| 戻り値 |  |
| :--- | :--- |
|  `Team` オブジェクト。|

### `user`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L728-L748)

```python
user(
    username_or_email: str
) -> Optional['public.User']
```

ユーザー名またはメールアドレスからユーザーを返します。

注意: この関数はローカル管理者にのみ機能します。自分のユーザーオブジェクトを取得しようとしている場合は、`api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) ユーザー名またはメールアドレス |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクト または ユーザーが見つからなかった場合はNone |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L750-L764)

```python
users(
    username_or_email: str
) -> List['public.User']
```

部分的なユーザー名またはメールアドレスクエリからすべてのユーザーを返します。

注意: この関数はローカル管理者にのみ機能します。自分のユーザーオブジェクトを取得しようとしている場合は、`api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) 見つけたいユーザーの接頭辞または接尾辞 |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクトの配列 |

