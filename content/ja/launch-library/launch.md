---
title: Launch API
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch.py#L249-L331 >}}

W&B の Launch で実験を起動します。

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
|  `job` |  wandb.Job を指す文字列参照。例: wandb/test/my-job:latest |
|  `api` |  wandb.apis.internal の wandb Api のインスタンス。 |
|  `entry_point` |  プロジェクト内で実行するエントリーポイント。wandb の URI の場合は元の run で使われたエントリーポイント、Git リポジトリの URI の場合は main.py が既定です。 |
|  `version` |  Git ベースのプロジェクトでは、コミットハッシュまたはブランチ名。 |
|  `name` |  run を起動する際の run 名。 |
|  `resource` |  run の実行バックエンド。 |
|  `resource_args` |  リモートバックエンドに run を起動するためのリソース関連の引数。作成される Launch の設定では `resource_args` の下に保存されます。 |
|  `project` |  起動した run を送信する対象の Project。 |
|  `entity` |  起動した run を送信する対象の Entity。 |
|  `config` |  run の設定を含む 辞書。キー "resource_args" の下にリソース固有の引数を含めることもできます。 |
|  `synchronous` |  run の完了を待つ間ブロックするかどうか。既定は True。`synchronous` が False で、`backend` が "local-container" の場合、このメソッドは復帰しますが、現在の プロセス はローカルの run が完了するまで終了時にブロックされます。現在の プロセス が中断された場合、このメソッドで起動されたすべての非同期 run は終了されます。`synchronous` が True で run が失敗した場合、現在の プロセス もエラーで終了します。 |
|  `run_id` |  run の ID（最終的には :name: フィールドの代替）。 |
|  `repository` |  リモートレジストリ用のリポジトリパスの文字列名。 |

#### 例:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B の Project を実行し、再現可能な Docker 環境を作成します
# ローカルホスト上で
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
|  起動された run に関する情報（例: run ID）を提供する `wandb.launch.SubmittedRun` のインスタンス。 |

| 例外 |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` ブロッキングモードで起動された run が失敗した場合に発生。 |