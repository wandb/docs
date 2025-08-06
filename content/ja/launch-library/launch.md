---
title: 'launch api


  ローンチ API'
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch.py#L249-L331 >}}

W&B Launch 実験を実行します。

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
|  `job` |  wandb.Job への文字列参照 例: wandb/test/my-job:latest |
|  `api` |  wandb.apis.internal からの wandb Api インスタンス |
|  `entry_point` |  プロジェクト内で実行するエントリポイント。デフォルトでは wandb URI の場合は元 run で使用されたエントリポイント、Git リポジトリ URI の場合は main.py が使用されます。 |
|  `version` |  Git ベースのプロジェクトの場合、コミットハッシュまたはブランチ名。 |
|  `name` |  実行する run の名前。 |
|  `resource` |  run を実行するための実行バックエンド。 |
|  `resource_args` |  リモートバックエンドで run を実行するためのリソース関連引数。作成される launch 設定の `resource_args` に保存されます。 |
|  `project` |  実行した run を送信する対象の Project |
|  `entity` |  実行した run を送信する対象の Entity |
|  `config` |  run 用の設定を含む辞書。リソース固有の引数を "resource_args" のキーで持つ場合もあります。 |
|  `synchronous` |  run の完了を待つ間、ブロックするかどうか。デフォルトは True。`synchronous` が False で `backend` が "local-container" の場合、このメソッドは返りますが、現在のプロセスはローカル run が終了するまで終了時にブロックされます。プロセスが中断された場合、このメソッドを通じて非同期で開始された run は終了されます。`synchronous` が True で run が失敗した場合は、現在のプロセスもエラーとなります。|
|  `run_id` |  run の ID（最終的に :name: フィールドを置き換えます）|
|  `repository` |  リモートレジストリのリポジトリパス名の文字列 |

#### 例:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B プロジェクトをローカルホスト上で再現可能な Docker 環境として実行
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
|  launch された run に関する情報（例：run IDなど）を持つ `wandb.launch.SubmittedRun` インスタンス。 |

| 例外 |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` ブロッキング モードで launch した run が失敗した場合に発生します。 |