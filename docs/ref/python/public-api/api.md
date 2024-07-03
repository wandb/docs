# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L95-L1179' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&B サーバーにクエリを送るために使用されます。

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
|  `overrides` |  (dict) `base_url` を設定します。他の `entity`, `project`, `run` のデフォルトも設定できます。|

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
|  `name` |  (str) artifact 名。`entity/project`でプレフィックスされる場合があります。以下の形式が有効です: name:version name:alias |
|  `type` |  (str, optional) 取得するartifactの種類。|

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。|

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L967-L983)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

タイプと `entity/project/name` の形式でパスを解析して単一のartifact collectionを返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するartifact collectionの種類。|
|  `name` |  (str) artifact collection 名。`entity/project`でプレフィックスされる場合があります。|

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactCollection` オブジェクト。|

### `artifact_collection_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1162-L1179)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

指定したprojectとentityの中でartifact collectionが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) artifact collection 名。`entity/project`でプレフィックスされる場合があります。entityまたはprojectが指定されていない場合、overrideパラメーターから推測されます。存在しない場合は、entityはユーザー設定から取得され、projectは「未分類」にデフォルト設定されます。|
|  `type` |  (str) artifact collectionの種類。|

| 戻り値 |  |
| :--- | :--- |
|  artifact collectionが存在する場合は True、そうでない場合は False。|

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L947-L965)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) -> "public.ArtifactCollections"
```

一致するartifact collectionsのコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project_name` |  (str) フィルターをかけるprojectの名前。|
|  `type_name` |  (str) フィルターをかけるartifactの種類。|
|  `per_page` |  (int, optional) クエリのページネーションのページサイズを設定します。 None を設定するとデフォルトのサイズが使用されます。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  イテラブルな `ArtifactCollections` オブジェクト。|

### `artifact_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1140-L1160)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

指定したprojectとentityの中でartifactのバージョンが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) artifact 名。`entity/project`でプレフィックスされる場合があります。entityまたはprojectが指定されていない場合、overrideパラメーターから推測されます。存在しない場合は、entityはユーザー設定から取得され、projectは「未分類」にデフォルト設定されます。以下の形式が有効です: name:version name:alias |
|  `type` |  (str, optional) artifactの種類。|

| 戻り値 |  |
| :--- | :--- |
|  artifactバージョンが存在する場合は True、そうでない場合は False。|

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L931-L945)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

一致する `ArtifactType` を返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するartifactの種類の名前。|
|  `project` |  (str, optional) 指定された場合、フィルターをかけるprojectの名前またはパス。|

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

一致するartifactの種類のコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project` |  (str, optional) 指定された場合、フィルターをかけるprojectの名前またはパス。|

| 戻り値 |  |
| :--- | :--- |
|  イテラブルな `ArtifactTypes` オブジェクト。|

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L985-L995)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

廃止予定です。代わりに `artifacts(type_name, name)` を使用してください。

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L997-L1015)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: Optional[int] = 50
) -> "public.Artifacts"
```

指定した引数に基づく `Artifacts` コレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得するartifactの種類。|
|  `name` |  (str) artifact collection 名。`entity/project`でプレフィックスされる場合があります。|
|  `per_page` |  (int, optional) クエリのページネーションのページサイズを設定します。 None を設定するとデフォルトのサイズが使用されます。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  イテラブルな `Artifacts` オブジェクト。|

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L281-L288)

```python
create_project(
    name: str,
    entity: str
) -> None
```

新規プロジェクトを作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 新しいprojectの名前。|
|  `entity` |  (str) 新しいprojectのentity。|

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

新規runを作成します。

| 引数 |  |
| :--- | :--- |
|  `run_id` |  (str, optional) 付与するrunのID。デフォルトではrun IDは自動生成されるため、通常は指定する必要はなく、指定するときは自己責任で行ってください。|
|  `project` |  (str, optional) 指定された場合、新しいrunのproject。|
|  `entity` |  (str, optional) 指定された場合、新しいrunのentity。|

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `Run`。|

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

新しいrun queue (launch) を作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 作成するキューの名前。|
|  `type` |  (str) キューに使用されるリソースの種類。"local-container", "local-process", "kubernetes", "sagemaker", "gcp-vertex" のいずれか。|
|  `entity` |  (str, optional) キューを作成するentityの名前。Noneの場合、設定済みまたはデフォルトのentityが使用されます。|
|  `prioritization_mode` |  (str, optional) 使用する優先順位付けのバージョン。"V0" または None|
|  `config` |  (dict, optional) キューに使用するデフォルトのリソース設定。ハンドルバー (例: "{{var}}") を使用してテンプレート変数を指定します。|
|  `template_variables` |  (dict) テンプレート変数スキーマの辞書、設定とともに使用します。形式は以下の通り: { "var-name": { "schema": { "type": ("string", "number", または "integer"), "default": (オプションの値), "minimum": (オプションの最小), "maximum": (オプションの最大), "enum": [..."(options)"] } } } |

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `RunQueue`|

| Raises |  |
| :--- | :--- |
|  引数が無効な場合は ValueError。 W&B APIエラーの場合は wandb.Error。|

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L705-L715)

```python
create_team(
    team, admin_username=None
)
```

新規チームを作成します。

| 引数 |  |
| :--- | :--- |
|  `team` |  (str) チームの名前。|
|  `admin_username` |  (str, optional) チームの管理ユーザーのユーザー名、デフォルトは現在のユーザー。|

| 戻り値 |  |
| :--- | :--- |
|  `Team` オブジェクト。|

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L424-L434)

```python
create_user(
    email, admin=(False)
)
```

新規ユーザーを作成します。

| 引数 |  |
| :--- | :--- |
|  `email` |  (str) ユーザのメールアドレス。|
|  `admin` |  (bool) このユーザーをグローバルインスタンス管理者にするか。|

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクト。|

### `flush`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L501-L508)

```python
flush()
```

ローカルキャッシュをフラッシュします。

apiオブジェクトはrunのローカルキャッシュを保持するため、スクリプトを実行している間にrunの状態が変わる場合は、最新の値を取得するために`api.flush()`でローカルキャッシュをクリアする必要があります。

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L510-L564)

```python
from_path(
    path
)
```

パスからproject、run、sweepまたはreportを返します。

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
|  `path` |  (str) project、run、sweepまたはreportへのパス。|

| 戻り値 |  |
| :--- | :--- |
|  `Project`、`Run`、`Sweep`、または `BetaReport` インスタンス。|

| Raises |  |
| :--- | :--- |
|  パスが無効だったり、オブジェクトが存在しない場合は wandb.Error。|

### `job`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1043-L1060)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

指定した引数に基づく `Job` を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) ジョブの名前。|
|  `path` |  (str, optional) ジョブアーティファクトをダウンロードするルートパス。|

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

指定したentityとprojectのジョブをリストで返します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str) ジョブのentity。|
|  `project` |  (str) ジョブのproject。|

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

指定された名前（およびoptionalでentity）の `Project` を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) projectの名前。|
|  `entity` |  (str) リクエストするentityの名前。Noneの場合、`Api` に渡されたデフォルトのentityが使用されます。デフォルトのentityがない場合、`ValueError` が発生します。|

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

指定したentityのprojectsを取得します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str, optional) リクエストするentityの名前。Noneの場合、`Api` に渡されたデフォルトのentityが使用されます。デフォルトのentityがない場合、`ValueError` が発生します。|
|  `per_page` |  (int, optional) クエリのページネーションのページサイズを設定します。Noneの場合、デフォルトのサイズが使用されます。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  `Projects` オブジェクト（`Project` オブジェクトのイテラブルコレクション）。|

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L863-L884)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

パスに基づいた単一のqueued runを返します。

`entity/project/queue_id/run_queue_item_id` の形式でパスを解析します。

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L672-L703)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) -> "public.Reports"
```

指定したproject pathのreportsを取得します。

警告: このAPIはベータ版であり、将来のリリースで変更される予定です。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) レポートが存在するプロジェクトへのパス。形式は "entity/project" である必要があります。|
|  `name` |  (str, optional) リクエストするレポートの自身の名前。|
|  `per_page` |  (int, optional) クエリのページネーションのページサイズを設定します。Noneの場合、デフォルトのサイズが使用されます。通常、これを変更する理由はありません。|

| 戻り値 |  |
| :--- | :--- |
|  `Reports` オブジェクト（`BetaReport` オブジェクトのイテラブルコレクション）。|

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L846-L861)

```python
run(
    path=""
)
```

`entity/project/run_id` の形式でパスを解析して単一のrunを返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) `entity/project/run_id` の形式のrunへのパス。`api.entity` が設定されている場合、この形式は `project/run_id` でも構いませんし、`api.project` が設定されている場合、この形式はrun_idだけでも構いません。|

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

指定されたentityの指定された名前の `RunQueue` を返します。

新しい `RunQueue` を作成するには、`wandb.Api().create_run_queue(...)`を使用します。

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

指定されたフィルターに一致するprojectから一連のrunsを返します。

`config.*`、`summary_metrics.*`、`tags`、`state`、`entity`、`createdAt` などでフィルターすることができます。

#### 例:

config.experiment_nameが "foo" に設定されているmy_projectのrunsを検索します。

```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

config.experiment_nameが "foo" または "bar" に設定されているmy_projectのrunsを検索します。

```
api.runs(
    path="my_entity/my_project",
    filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```

config.experiment_nameが正規表現に一致するmy_projectのrunsを検索します（アンカーはサポートされていません）。

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

run名が正規表現に一致するmy_projectのrunsを検索します（アンカーはサポートされていません）。

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}}
)
```

昇順で損失を並べ替えたmy_projectのrunsを検索します。

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) プロジェクトへのパス。形式は "entity/project" である必要があります。|
|  `filters` |  (dict) MongoDBクエリ言語を使用して、特定のrunsをクエリします。config.key、summary_metrics.key、state、entity、createdAt などのrunプロパティでフィルターします。例えば、{"config.experiment_name": "foo"} はexperiment_nameが "foo" に設定されているrunsを検索します。より複雑なクエリを作成するために操作を組み合わせることができます。言語に関するリファレンスは以下を参照してください: https://docs.mongodb.com/manual/reference/operator/query |
|  `order` |  (str) 順序の対象となるプロパティ。`created_at`、`heartbeat_at`、`config.*.value`、または `summary_metrics.*` を指定します。順序の前に `+` を付けると昇順になり、`-` を付けると降順になります（デフォルト）。デフォルトの順序は、最新から古い順へのrun.created_at です。|
|  `per_page` |  (int) クエリのページネーションのページサイズを設定します。|
|  `include_sweeps` |  (bool) 結果にsweep runsを含めるかどうか。|

| 戻り値 |  |
| :--- | :--- |
|  `Runs` オブジェクト（`Run` オブジェクトのイテラブルコレクション）。|

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L901-L916)

```python
sweep(
    path=""
)
```

`entity/project/sweep_id` の形式でパスを解析して単一のsweepを返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str, optional) `entity/project/sweep_id` の形式のsweepへのパス。`api.entity` が設定されている場合、この形式はproject/sweep_id でも構いませんし、`api.project` が設定されている場合、この形式はsweep_idだけでも構いません。|

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

tfeventファイルを含むローカルディレクトリをW&Bと同期します。

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L717-L726)

```python
team(
    team: str
) -> "public.Team"
```

指定された名前の `Team` を返します。

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

注: この関数はローカル管理者のみに機能します。自身のユーザーオブジェクトを取得する場合は `api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) ユーザーのユーザー名またはメールアドレス。|

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクト、またはユーザーが見つからない場合は None。|

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L750-L764)

```python
users(
    username_or_email: str
) -> List['public.User']
```

部分的なユーザー名またはメールアドレスクエリからすべてのユーザーを返します。

注: この関数はローカル管理者のみに機能します。自身のユーザーオブジェクトを取得する場合は `api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) 検索するユーザーのプレフィックスまたはサフィックス。|

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクトの配列。|

| クラス変数 |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `DEFAULT_ENTITY_QUERY`<a id="DEFAULT_ENTITY_QUERY"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |