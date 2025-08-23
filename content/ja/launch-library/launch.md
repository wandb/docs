---
title: ローンチ API
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch.py#L249-L331 >}}

W&B ローンチ実験を開始します。

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

| 引数 | 説明 |
| :--- | :--- |
|  `job` |  wandb.Job への文字列参照（例: wandb/test/my-job:latest）|
|  `api` |  wandb.apis.internal から取得した wandb Api のインスタンス |
|  `entry_point` |  プロジェクト内で実行するエントリーポイント。デフォルトでは wandb URI では元の run で使用されたエントリーポイントを、git リポジトリ URI では main.py を利用します。|
|  `version` |  Git ベースのプロジェクトの場合はコミットハッシュまたはブランチ名。|
|  `name` |  ローンチする run の名前。|
|  `resource` |  run を実行するバックエンド。|
|  `resource_args` |  リモートバックエンドでの run 実行時に必要なリソース関連の引数。作成されるローンチの config 内の `resource_args` に保存されます。|
|  `project` |  ローンチした run を送信する対象 Project |
|  `entity` |  ローンチした run を送信する対象 Entity |
|  `config` |  run の設定を含む辞書。resource 特有の引数を "resource_args" キーで含めることもできます。|
|  `synchronous` |  run の完了を待つかどうか（ブロックするか）。デフォルトは True。`synchronous` が False かつ `backend` が "local-container" の場合はこのメソッドの返却後、ローカル run 完了までプロセス終了時にブロックします。プロセスが中断されると非同期実行された run も終了します。`synchronous` が True で run が失敗した場合、現在のプロセスもエラーになります。|
|  `run_id` |  run の ID（最終的には :name: フィールドを置き換える予定）|
|  `repository` |  リモートレジストリ用のリポジトリパスの文字列名 |

#### 利用例：

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B Project をローカルホスト上で再現可能な Docker 環境として実行
# 実行
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 戻り値 | 説明 |
| :--- | :--- |
|  ローンチした run の情報（例: run ID など）へアクセスできる `wandb.launch.SubmittedRun` のインスタンス。|

| 例外 | 説明 |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` ブロッキングモードで実行された run が失敗した場合発生します。|