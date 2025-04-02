---
title: launch
menu:
  reference:
    identifier: ja-ref-python-launch-library-launch
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/_launch.py#L249-L331 >}}

W&B の Launch の experiment を Launch します。

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
|  `job` |  wandb.Job への文字列参照 (例: wandb/test/my-job:latest) |
|  `api` |  wandb.apis.internal からの wandb Api のインスタンス。 |
|  `entry_point` |  プロジェクト内で実行するエントリポイント。wandb URI の場合は元の run で使用されたエントリポイント、git リポジトリ URI の場合は main.py をデフォルトで使用します。 |
|  `version` |  Git ベースのプロジェクトの場合、コミットハッシュまたはブランチ名のいずれか。 |
|  `name` |  run を Launch する run 名。 |
|  `resource` |  run の実行バックエンド。 |
|  `resource_args` |  リモートバックエンドに run を Launch するためのリソース関連の引数。`resource_args` の下の構築された Launch 設定に保存されます。 |
|  `project` |  Launch された run の送信先となる対象の Project |
|  `entity` |  Launch された run の送信先となる対象の Entity |
|  `config` |  run の設定を含む辞書。キー "resource_args" の下にあるリソース固有の引数も含む場合があります。 |
|  `synchronous` |  run の完了を待機中にブロックするかどうか。デフォルトは True です。`synchronous` が False で、`backend` が "local-container" の場合、このメソッドは戻りますが、現在のプロセスはローカル run が完了するまで終了時にブロックされます。現在のプロセスが中断された場合、このメソッドを介して Launch された非同期 run はすべて終了します。`synchronous` が True で、run が失敗した場合、現在のプロセスもエラーになります。 |
|  `run_id` |  run の ID (最終的に :name: フィールドを置き換えるため) |
|  `repository` |  リモートレジストリのリポジトリパスの文字列名 |

#### 例:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B プロジェクトを実行し、再現可能な Docker 環境をローカルホスト上に作成します。
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
|  Launch された run に関する情報 (run ID など) を公開する `wandb.launch.SubmittedRun` のインスタンス。 |

| 例外 |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` ブロックモードで Launch された run が失敗した場合。 |
