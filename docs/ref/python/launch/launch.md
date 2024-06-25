
# launch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/_launch.py#L246-L328' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&B launch 実験を起動します。

```python
launch(
    api: Api,
    job: Optional[str] = None,
    entry_point: Optional[List[str]] = None,
    version: Optional[str] = None,
    name: Optional[str] = None,
    resource: Optional[str] = None,
    resource_args: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    docker_image: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    synchronous: Optional[bool] = (True),
    run_id: Optional[str] = None,
    repository: Optional[str] = None
) -> AbstractRun
```

| 引数 |  |
| :--- | :--- |
|  `job` |  wandb.Job への文字列参照例: wandb/test/my-job:latest |
|  `api` |  wandb.apis.internal からの wandb Api のインスタンス。 |
|  `entry_point` |  Project 内で実行するエントリーポイント。デフォルトは、wandb URI の場合はオリジナルの run に使用されたエントリーポイント、git リポジトリ URI の場合は main.py。 |
|  `version` |  Git ベースの Projects の場合、コミットハッシュまたはブランチ名。 |
|  `name` |  run を起動する際の名前。 |
|  `resource` |  run の実行バックエンド。 |
|  `resource_args` |  リモートバックエンドに実行を送信するためのリソース関連の引数。この引数は構築された launch 設定内の `resource_args` に格納されます。 |
|  `project` |  起動した run を送信する対象 Project |
|  `entity` |  起動した run を送信する対象 Entity |
|  `config` |  run の設定を含む辞書。キー "resource_args" でリソース固有の引数も含むことができます。 |
|  `synchronous` |  run が完了するまで待機するかどうかを指定します。デフォルトは True です。`synchronous` が False で `backend` が "local-container" の場合、このメソッドは戻りますが、現在のプロセスはローカルの run が完了するまで終了時にブロックします。もし現在のプロセスが中断された場合、このメソッド経由で起動された非同期 run は全て終了されます。`synchronous` が True で run が失敗した場合、現在のプロセスもエラーを出します。 |
|  `run_id` |  run の ID（最終的には :name: フィールドを置き換えるため） |
|  `repository` |  リモートレジストリのリポジトリパスの文字列名 |

#### 例:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B Project を実行し、再現可能な Docker 環境をローカルホストに作成
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
|  起動された run に関する情報（例: run ID）を公開する `wandb.launch.SubmittedRun` のインスタンス。 |

| 例外 |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` ブロッキングモードで起動された run が成功しなかった場合。 |