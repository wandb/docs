
# launch_add

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/_launch_add.py#L34-L131' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&BのLaunch experimentをエンキューする。source uri、job、またはdocker_imageのいずれかを使用してください。

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

| 引数                  |                                                                                                                                                                     |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uri`                | 実行する実験のURI。wandb runのURIまたはGitリポジトリのURI。                                                                                                        |
| `job`                | wandb.Jobへの文字列参照 例: wandb/test/my-job:latest                                                                                                               |
| `config`             | runの設定を含む辞書。キー"resource_args"の下にリソース固有の引数も含めることができます。                                                                          |
| `template_variables` | runキューのテンプレート変数の値を含む辞書。形式は{`"<変数名>"`: `<変数値>`}を期待します。                                                                            |
| `project`            | 起動したrunを送信するターゲットプロジェクト                                                                                                                        |
| `entity`             | 起動したrunを送信するターゲットエンティティ                                                                                                                       |
| `queue`              | runをenqueueするキューの名前                                                                                                                                       |
| `priority`           | ジョブの優先度レベル。1が最も高い優先度                                                                                                                            |
| `resource`           | runの実行バックエンド。W&Bは「local-container」バックエンドのサポートを提供します。                                                                                 |
| `entry_point`        | プロジェクト内で実行するエントリーポイント。デフォルトはwandb URIの場合はオリジナルrunで使用されたエントリーポイント、Gitリポジトリ URIの場合はmain.pyを使用します。  |
| `name`               | runを実行する際の名前。                                                                                                                                            |
| `version`            | Gitベースのプロジェクトの場合、コミットハッシュまたはブランチ名。                                                                                                  |
| `docker_image`       | runに使用するdockerイメージの名前。                                                                                                                               |
| `resource_args`      | リモートバックエンドにrunを起動するためのリソース関連引数。構築されたlaunch configの`resource_args`の下に保存されます。                                          |
| `run_id`             | 起動したrunのIDを示すオプションの文字列。                                                                                                                          |
| `build`              | デフォルトfalseのオプションフラグ。buildの場合、イメージが作成され、ジョブアーティファクトを生成し、そのジョブアーティファクトへの参照をキューにプッシュします。     |
| `repository`         | レジストリにイメージをプッシュする際に使用されるリモートリポジトリの名前を制御するオプションの文字列。                                                               |
| `project_queue`      | キューのプロジェクト名を制御するオプションの文字列。主にプロジェクトスコープのキューとの互換性のために使用されます。                                            |

#### 例:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&Bプロジェクトを実行し、ローカルホスト上に再現可能なdocker環境を作成
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 戻り値                                                                                                                                                                                    |     |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-- |
| `wandb.api.public.QueuedRun`のインスタンスを返します。これは、queued runに関する情報を提供し、`wait_until_started`または`wait_until_finished`が呼ばれた場合、基礎となるRun情報へのアクセスを提供します。 |

| 例外                                                      |     |
| :------------------------------------------------------- | :-- |
| `wandb.exceptions.LaunchError`が発生した場合、失敗を示します。 |