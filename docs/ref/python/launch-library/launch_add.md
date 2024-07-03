# launch_add

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/launch/_launch_add.py#L34-L131' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

ソースの URI、ジョブ、または docker_image を使って、W&B launch 実験をキューに追加します。

```python
launch_add(
    uri: Optional[str] = None,
    job: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    template_variables: Optional[Dict[str, Union[float, int, str]]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    queue_name: Optional[str] = None,
    resource: Optional[str] = None,
    entry_point: Optional[List[str]] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    docker_image: Optional[str] = None,
    project_queue: Optional[str] = None,
    resource_args: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    build: Optional[bool] = (False),
    repository: Optional[str] = None,
    sweep_id: Optional[str] = None,
    author: Optional[str] = None,
    priority: Optional[int] = None
) -> "public.QueuedRun"
```

| 引数 |  |
| :--- | :--- |
|  `uri` |  実行する実験の URI。wandb run の URI または Git リポジトリの URI。 |
|  `job` |  wandb.Job への文字列参照。例: wandb/test/my-job:latest |
|  `config` |  run の設定を含む辞書。リソース固有の引数を `resource_args` キーの下に含むこともできます。 |
|  `template_variables` |  キューのテンプレート変数の値を含む辞書。期待される形式は {"VAR_NAME": VAR_VALUE} |
|  `project` |  launch された run を送信するターゲット Project |
|  `entity` |  launch された run を送信するターゲット Entity |
|  `queue` |  run をキューに追加するためのキューの名前 |
|  `priority` |  ジョブの優先度レベル。1 が最も高い優先度 |
|  `resource` |  run の実行バックエンド: W&B は "local-container" バックエンドの組み込みサポートを提供します |
|  `entry_point` |  プロジェクト内で実行するエントリーポイント。デフォルトでは wandb URIs の場合は元の run で使用されたエントリーポイント、Git リポジトリ URIs の場合は main.py を使用します。 |
|  `name` |  run の名前。 |
|  `version` |  Git に基づく Projects の場合、コミットハッシュまたはブランチ名。 |
|  `docker_image` |  run に使用する docker イメージの名前。 |
|  `resource_args` |  リモートバックエンドでの run を launch するためのリソース関連の引数。生成された launch 設定の `resource_args` の下に保存されます。 |
|  `run_id` |  launch された run の ID を示すオプションの文字列 |
|  `build` |  オプションのフラグ。デフォルトは false。キューが設定されている場合、イメージが作成され、ジョブアーティファクトが生成され、そのジョブアーティファクトへの参照がキューにプッシュされます。 |
|  `repository` |  イメージをレジストリにプッシュする際に使用されるリモートリポジトリの名前を制御するオプションの文字列 |
|  `project_queue` |  キューのためのプロジェクトの名前を制御するオプションの文字列。主にプロジェクトスコープのキューとの互換性のために使用されます。 |

#### 例:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B プロジェクトを実行し、再現可能な Docker 環境を
# ローカルホストで作成する
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
|  `wandb.api.public.QueuedRun` のインスタンス。キューに追加された run に関する情報を提供し、`wait_until_started` や `wait_until_finished` が呼ばれた場合、基礎となる Run の情報にアクセスすることができます。 |

| 発生する例外 |  |
| :--- | :--- |
|  `wandb.exceptions.LaunchError` が発生した場合 |
