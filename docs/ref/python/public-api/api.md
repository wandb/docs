
# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L95-L1179' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

wandb サーバーにクエリを送るために使用されます。

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
|  `overrides` |  (辞書) `base_url` を、https://api.wandb.ai 以外の wandb サーバーを使用している場合に設定できます。また、`entity`、`project`、および `run` のデフォルトを設定することもできます。 |

| 属性 |  |
| :--- | :--- |

## メソッド

### `artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L1017-L1041)

```python
artifact(
    name, type=None
)
```

`entity/project/name` の形式で指定されたパスを解析して、単一の artifact を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) artifactの名前。`entity/project`で接頭辞を付けることができます。有効な名前は次の形式である必要があります: name:version または name:alias |
|  `type` |  (str, 任意) 取得する artifact の種類。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `artifact_collection`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L967-L983)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

`entity/project/name` の形式で指定されたパスを解析して、単一の artifact collection を返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得する artifact collection の種類。 |
|  `name` |  (str) artifact collection の名前。`entity/project`で接頭辞を付けることができます。 |

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactCollection` オブジェクト。 |

### `artifact_collection_exists`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L1162-L1179)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

指定された project と entity 内で artifact collection が存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) artifact collection の名前。`entity/project`で接頭辞を付けることができます。entity または project が指定されていない場合、上書きパラメータから推測されます。そうでない場合、entity はユーザー設定から取得され、project は "uncategorized" に設定されます。 |
|  `type` |  (str) artifact collection の種類。 |

| 戻り値 |  |
| :--- | :--- |
|  artifact collection が存在する場合は True、そうでない場合は False。 |

### `artifact_collections`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L947-L965)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) -> "public.ArtifactCollections"
```

一致する artifact collections のコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project_name` |  (str) フィルターするプロジェクトの名前。 |
|  `type_name` |  (str) フィルターする artifact の種類の名前。 |
|  `per_page` |  (int, 任意) クエリのページネーションのページサイズを設定します。None はデフォルトサイズを使用します。通常、この値を変更する理由はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactCollections` オブジェクトの反復可能なもの。 |

### `artifact_exists`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L1140-L1160)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

指定された project と entity 内で artifact のバージョンが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) artifact の名前。`entity/project`で接頭辞を付けることができます。entity または project が指定されていない場合、上書きパラメータから推測されます。そうでない場合、entity はユーザー設定から取得され、project は "uncategorized" に設定されます。有効な名前は次の形式である必要があります: name:version または name:alias |
|  `type` |  (str, 任意) artifact の種類。 |

| 戻り値 |  |
| :--- | :--- |
|  artifact のバージョンが存在する場合は True、そうでない場合は False。 |

### `artifact_type`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L931-L945)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

一致する `ArtifactType` を返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (str) 取得する artifact の種類の名前。 |
|  `project` |  (str, 任意) 指定された場合、フィルター対象のプロジェクトの名前またはパス。 |

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactType` オブジェクト。 |

### `artifact_types`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L918-L929)

```python
artifact_types(
    project: Optional[str] = None
) -> "public.ArtifactTypes"
```

一致する artifact タイプのコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project` |  (str, 任意) 指定された場合、フィルター対象のプロジェクトの名前またはパス。 |

| 戻り値 |  |
| :--- | :--- |
|  `ArtifactTypes` オブジェクトの反復可能なもの。 |

### `artifact_versions`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L985-L995)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

廃止予定、 代わりに `artifacts(type_name, name)` を使用してください。

### `artifacts`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L997-L1015)

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
|  `type_name` |  (str) 取得する artifact の種類。 |
|  `name` |  (str) artifact collection の名前。`entity/project`で接頭辞を付けることができます。 |
|  `per_page` |  (int, 任意) クエリのページネーションのページサイズを設定します。None はデフォルトサイズを使用します。通常、この値を変更する理由はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifacts` オブジェクトの反復可能なもの。 |

### `create_project`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L281-L288)

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
|  `entity` |  (str) 新しいプロジェクトの entity。 |

### `create_run`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L290-L310)

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
|  `run_id` |  (str, 任意) 付与する run の ID。run ID はデフォルトで自動生成されるため、一般的にこれを指定する必要はなく、指定する場合は自己責任で行います。 |
|  `project` |  (str, 任意) 指定された場合、新しい run のプロジェクト。 |
|  `entity` |  (str, 任意) 指定された場合、新しい run の entity。 |

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `Run`。 |

### `create_run_queue`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L312-L422)

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

新しい run キュー（launch）を作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) 作成するキューの名前 |
|  `type` |  (str) キューのために使用されるリソースの種類。"local-container"、"local-process"、"kubernetes"、"sagemaker"、または "gcp-vertex" のいずれか。 |
|  `entity` |  (str) キューの作成先の entity の名前。None の場合は設定済みの entity またはデフォルトの entity を使用します。 |
|  `prioritization_mode` |  (str) 優先順位付けに使用するバージョン。"V0" または None のいずれか。 |
|  `config` |  (辞書) キューに使用するデフォルトのリソース設定。ハンドルバー（例: "{{var}}"）を使用してテンプレート変数を指定します。 |
|  `template_variables` |  (辞書) 設定と共に使用するテンプレート変数のスキーマの辞書。期待される形式は次の通り: { "var-name": { "schema": { "type": ("string", "number", または "integer")、"default": (任意の値)、"minimum": (任意の最小値)、"maximum": (任意の最大値)、"enum": [..."(選択肢)"] } } } |

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `RunQueue` |

| 例外 |  |
| :--- | :--- |
|  パラメータのいずれかが無効な場合は ValueError、wandb API エラーが発生した場合は wandb.Error |

### `create_team`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L705-L715)

```python
create_team(
    team, admin_username=None
)
```

新しいチームを作成します。

| 引数 |  |
| :--- | :--- |
|  `team` |  (str) チームの名前 |
|  `admin_username` |  (str) チームの管理者ユーザーの名前。デフォルトは現在のユーザーです。 |

| 戻り値 |  |
| :--- | :--- |
|  `Team` オブジェクト。 |

### `create_user`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L424-L434)

```python
create_user(
    email, admin=(False)
)
```

新しいユーザーを作成します。

| 引数 |  |
| :--- | :--- |
|  `email` |  (str) ユーザーのメールアドレス。 |
|  `admin` |  (bool) このユーザーをグローバルインスタンス管理者にするかどうか。 |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクト。 |

### `flush`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L501-L508)

```python
flush()
```

ローカルキャッシュをフラッシュします。

Api オブジェクトは runs のローカルキャッシュを保持するため、スクリプトを実行中に run の状態が変わる可能性がある場合は、`api.flush()` を使用して run に関連付けられた最新の値を取得する必要があります。

### `from_path`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L510-L564)

```python
from_path(
    path
)
```

パスから run, sweep, project または report を返します。

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
|  `path` |  (str) プロジェクト、run、sweep、または report へのパス。 |

| 戻り値 |  |
| :--- | :--- |
|  `Project`、 `Run`、 `Sweep`、または `BetaReport` インスタンス。 |

| 例外 |  |
| :--- | :--- |
|  path が無効またはオブジェクトが存在しない場合は wandb.Error |

### `job`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L1043-L1060)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

指定されたパラメータから `Job` を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) Job の名前。 |
|  `path` |  (str, 任意) 指定された場合、Job artifact をダウンロードするルートパス。 |

| 戻り値 |  |
| :--- | :--- |
|  `Job` オブジェクト。 |

### `list_jobs`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L1062-L1138)

```python
list_jobs(
    entity: str,
    project: str
) -> List[Dict[str, Any]]
```

指定された entity と project に対する job のリストを返します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str) リストされた job(s) の entity。 |
|  `project` |  (str) リストされた job(s) の project。 |

| 戻り値 |  |
| :--- | :--- |
|  一致する job のリスト。 |

### `project`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L657-L670)

```python
project(
    name: str,
    entity: Optional[str] = None
) -> "public.Project"
```

指定された名前（および提供されていれば entity）の `Project` を返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) プロジェクト名。 |
|  `entity` |  (str) リクエストされた entity の名前。None の場合、デフォルトの entity が `Api` に渡されます。デフォルトの entity がない場合は `ValueError` を発生させます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Project` オブジェクト。 |

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L631-L655)

```python
projects(
    entity: Optional[str] = None,
    per_page: Optional[int] = 200
) -> "public.Projects"
```

指定された entity のプロジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (str) リクエストされた entity の名前。None の場合、デフォルトの entity が `Api` に渡されます。デフォルトの entity がない場合は `ValueError` を発生させます。 |
|  `per_page` |  (int) クエリペジネーションのページサイズを設定します。None の場合、デフォルトのサイズが使用されます。通常この値を変更する理由はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  `Projects` オブジェクト。これは `Project` オブジェクトの反復可能なコレクションです。 |

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L863-L884)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

指定されたパスに基づいて単一のキューに入れられた run を返します。

entity/project/queue_id/run_queue_item_id の形式のパスを解析します。

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L672-L703)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) -> "public.Reports"
```

指定されたプロジェクトパスのレポートを取得します。

警告: このAPIはベータ版であり、将来のリリースで変更される可能性があります。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) レポートが存在するプロジェクトのパス。形式は "entity/project" である必要があります。 |
|  `name` |  (str, optional) リクエストされたレポートのオプション名。 |
|  `per_page` |  (int) クエリペジネーションのページサイズを設定します。None の場合、デフォルトのサイズが使用されます。通常この値を変更する理由はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  `Reports` オブジェクト。これは `BetaReport` オブジェクトの反復可能なコレクションです。 |

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L846-L861)

```python
run(
    path=""
)
```

entity/project/run_id 形式のパスを解析して単一の run を返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) `entity/project/run_id` 形式の run のパス。`api.entity` が設定されている場合、これは `project/run_id` の形式にすることができ、`api.project` が設定されている場合、run_id だけにすることができます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Run` オブジェクト。 |

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L886-L899)

```python
run_queue(
    entity, name
)
```

指定された entity の名前付き `RunQueue` を返します。

新しい `RunQueue` を作成するには、 `wandb.Api().create_run_queue(...)` を使用します。

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L766-L844)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

提供されたフィルターに一致するプロジェクトから一連の runs を返します。

`config.*`、`summary_metrics.*`、`tags`、`state`、`entity`、`createdAt` などでフィルタリングできます。

#### 例:

config.experiment_name が "foo" に設定されている my_project 内の runs を検索

```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

config.experiment_name が "foo" または "bar" に設定されている my_project 内の runs を検索

```
api.runs(
    path="my_entity/my_project",
    filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```

config.experiment_name が正規表現に一致する my_project 内の runs を検索 (アンカーはサポートされていません)

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

run 名が正規表現に一致する my_project 内の runs を検索 (アンカーはサポートされていません)

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}}
)
```

損失が昇順になった my_project 内の runs を検索

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) プロジェクトへのパス。形式は "entity/project" です。 |
|  `filters` |  (dict) MongoDB クエリ言語を使用して特定の runs をクエリします。run のプロパティでフィルタリングできます。例えば config.key、summary_metrics.key、state、entity、createdAt など。例えば: {"config.experiment_name": "foo"} は、experiment_name の config エントリが "foo" に設定されている runs を検索します。操作を組み合わせてより複雑なクエリを作成することもできます。参考として、クエリ言語のドキュメントは https://docs.mongodb.com/manual/reference/operator/query にあります。 |
|  `order` |  (str) 順序は `created_at`、`heartbeat_at`、`config.*.value`、`summary_metrics.*` です。順序の前に + を付けると昇順に、- を付けると降順（デフォルト）になります。デフォルトの順序は run.created_at の新しい順から古い順です。 |
|  `per_page` |  (int) クエリペジネーションのページサイズを設定します。 |
|  `include_sweeps` |  (bool) 結果に sweep runs を含めるかどうか。 |

| 戻り値 |  |
| :--- | :--- |
|  `Runs` オブジェクト。これは `Run` オブジェクトの反復可能なコレクションです。 |

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L901-L916)

```python
sweep(
    path=""
)
```

`entity/project/sweep_id` 形式のパスを解析して sweep を返します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str, optional) `entity/project/sweep_id` 形式の sweep へのパス。`api.entity` が設定されている場合、これは project/sweep_id の形式にすることができ、`api.project` が設定されている場合、sweep_id だけにすることができます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Sweep` オブジェクト。 |

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L436-L458)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

tfevent ファイルを含むローカルディレクトリを wandb に同期します。

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L717-L726)

```python
team(
    team: str
) -> "public.Team"
```

指定された名前の `Team` を返します。

| 引数 |  |
| :--- | :--- |
|  `team` |  (str) チームの名前。 |

| 戻り値 |  |
| :--- | :--- |
|  `Team` オブジェクト。 |

### `user`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L728-L748)

```python
user(
    username_or_email: str
) -> Optional['public.User']
```

ユーザー名またはメールアドレスからユーザーを返します。

注: この機能はローカル管理者のみが使用できます。自分の user オブジェクトを取得しようとしている場合は、`api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) ユーザーの名前またはメールアドレス |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクトまたはユーザーが見つからなかった場合は None |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L750-L764)

```python
users(
    username_or_email: str
) -> List['public.User']
```

部分的なユーザー名またはメールアドレスクエリからすべてのユーザーを返します。

注: この機能はローカル管理者のみが使用できます。自分の user オブジェクトを取得しようとしている場合は、`api.viewer` を使用してください。

| 引数 |  |
| :--- | :--- |
|  `username_or_email` |  (str) 検索したいユーザーの接頭辞または接尾辞 |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクトの配列 |

| クラス変数 |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `DEFAULT_ENTITY_QUERY`<a id="DEFAULT_ENTITY_QUERY"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |