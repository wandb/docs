
# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L95-L1179' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースコードを見る</a></button></p>

wandbサーバーへクエリを実行するために使用されます。

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
|  `overrides` |  (辞書) https://api.wandb.ai 以外のwandbサーバーを使用している場合に `base_url` を設定できます。また、`entity`、`project`、`run` のデフォルトを設定することもできます。 |

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

`entity/project/name` の形式でパスを解析して単一のartifactを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (文字列) アーティファクト名。entity/projectで始まる場合もあります。有効な名前は以下の形式である可能性があります: name:version name:alias |
|  `type` |  (文字列, オプション) 取得するartifactのタイプ。 |

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

タイプと `entity/project/name` 形式のパスを解析して単一のartifactコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `type_name` |  (文字列) 取得するartifactコレクションのタイプ。 |
|  `name` |  (文字列) アーティファクトコレクション名。entity/projectで始まる場合もあります。 |

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

指定されたプロジェクトとエンティティ内にartifactコレクションが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (文字列) アーティファクトコレクション名。entity/projectで始まる場合もあります。entityまたはprojectが指定されていない場合、使用されるのは設定の値かデフォルトのentityとなります。 |
|  `type` |  (文字列) artifactコレクションのタイプ |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトコレクションが存在する場合はTrue、そうでない場合はFalse。 |

### `artifact_collections`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L947-L965)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) -> "public.ArtifactCollections"
```

条件に一致するartifactコレクションのコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project_name` |  (文字列) フィルターするプロジェクトの名前。 |
|  `type_name` |  (文字列) フィルターするartifactのタイプ名。 |
|  `per_page` |  (int, オプション) クエリのページネーションのためのページサイズを設定します。Noneはデフォルトのサイズを使用します。通常、この値を変更する必要はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  実行可能な `ArtifactCollections` オブジェクト。 |

### `artifact_exists`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L1140-L1160)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

指定されたプロジェクトとエンティティ内にartifactのバージョンが存在するかどうかを返します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (文字列) アーティファクト名。entity/projectで始まる場合もあります。entityまたはprojectが指定されていない場合、使用されるのは設定の値かデフォルトのentityとなります。有効な名前は次の形式である可能性があります: name:version name:alias |
|  `type` |  (文字列, オプション) アーティファクトのタイプ |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトバージョンが存在する場合はTrue、そうでない場合はFalse。 |

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
|  `type_name` |  (文字列) 取得するartifactタイプの名前。 |
|  `project` |  (文字列, オプション) フィルターするプロジェクト名またはパス。 |

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

一致するartifactタイプのコレクションを返します。

| 引数 |  |
| :--- | :--- |
|  `project` |  (文字列, オプション) フィルターするプロジェクト名またはパス。 |

| 戻り値 |  |
| :--- | :--- |
|  実行可能な `ArtifactTypes` オブジェクト。 |

### `artifact_versions`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L985-L995)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

非推奨、代わりに `artifacts(type_name, name)` を使用してください。

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
|  `type_name` |  (文字列) 取得するartifactのタイプ。 |
|  `name` |  (文字列) アーティファクトコレクション名。entity/projectで始まる場合があります。 |
|  `per_page` |  (int, オプション) クエリのページネーションのためのページサイズを設定します。Noneはデフォルトのサイズを使用します。通常、この値を変更する必要はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  実行可能な `Artifacts` オブジェクト。 |

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
|  `name` |  (文字列) 新しいプロジェクトの名前。 |
|  `entity` |  (文字列) 新しいプロジェクトのエンティティ。 |

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

新しいrunを作成します。

| 引数 |  |
| :--- | :--- |
|  `run_id` |  (文字列, オプション) runに割り当てるID。指定しなければ、自動的に生成されます。 |
|  `project` |  (文字列, オプション) 新しいrunのプロジェクト。 |
|  `entity` |  (文字列, オプション) 新しいrunのエンティティ。 |

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

新しいrunキュー (launch) を作成します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (文字列) 作成するキューの名前 |
|  `type` |  (文字列) キューに使用するリソースのタイプ。"local-container"、"local-process"、"kubernetes"、"sagemaker"、または "gcp-vertex" のいずれか。 |
|  `entity` |  (文字列) キューを作成するエンティティのオプション名。Noneの場合、構成済みのデフォルトエンティティを使用します。 |
|  `prioritization_mode` |  (文字列) 使用する優先順位モードのバージョン。 "V0" または None |
|  `config` |  (辞書) キューに使用するデフォルトのリソース設定。テンプレート変数を指定するにはハンドルバー (例: "{{var}}") を使用します。 |
|  `template_variables` |  (辞書) configで使用するテンプレート変数のスキーマ辞書。次の形式で期待されます: { "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (省略可能な値), "minimum": (省略可能な最小値), "maximum": (省略可能な最大値), "enum": [..."(選択肢)"] } } } |

| 戻り値 |  |
| :--- | :--- |
|  新しく作成された `RunQueue` |

| 例外 |  |
| :--- | :--- |
|  ValueError 無効なパラメータがある場合 wandb.Error wandb APIエラー |

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
|  `team` |  (文字列) チームの名前 |
|  `admin_username` |  (文字列) チームの管理者ユーザー名、デフォルトは現在のユーザー。 |

| 戻り値 |  |
| :--- | :--- |
|  `Team` オブジェクト |

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
|  `email` |  (文字列) ユーザーのメールアドレス |
|  `admin` |  (bool) このユーザーがグローバルインスタンス管理者かどうか |

| 戻り値 |  |
| :--- | :--- |
|  `User` オブジェクト |

### `flush`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L501-L508)

```python
flush()
```

ローカルキャッシュをフラッシュします。

apiオブジェクトは実行のローカルキャッシュを保持しますので、スクリプトの実行中に実行の状態が変更される可能性がある場合は、`api.flush()` を使用して run に関連する最新の値を取得する必要があります。

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
|  `path` |  (文字列) プロジェクト、 run、sweep または report へのパス |

| 戻り値 |  |
| :--- | :--- |
|  `Project`、`Run`、`Sweep`、または `BetaReport` インスタンス。 |

| 例外 |  |
| :--- | :--- |
|  wandb.Error パスが無効な場合やオブジェクトが存在しない場合 |

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
|  `name` |  (文字列) ジョブ名。 |
|  `path` |  (文字列, オプション) 指定された場合、ジョブアーティファクトをダウンロードするルートパス。 |

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

指定されたエンティティとプロジェクトのジョブのリストを返します。

| 引数 |  |
| :--- | :--- |
|  `entity` |  (文字列) リストされるジョブのエンティティ。 |
|  `project` |  (文字列) リストされるジョブのプロジェクト。 |

| 戻り値 |  |
| :--- | :--- |
|  一致するジョブのリスト。 |

### `project`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/api.py#L657-L670)

```python
project(
   