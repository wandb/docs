---
title: launch_add
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch_add.py#L34-L131 >}}

W&B Launch 実験をキューに追加します。source uri、job、または docker_image のいずれかが利用可能です。

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
|  `uri` |  実行する実験の URI。wandb run の uri または Git リポジトリの URI が指定できます。 |
|  `job` |  wandb.Job を指す文字列（例: wandb/test/my-job:latest） |
|  `config` |  run の設定を含む辞書型。`resource_args` というキーでリソース固有の引数も含めることができます。 |
|  `template_variables` |  run queue 用のテンプレート変数の値を含む辞書型。フォーマット例: `{"VAR_NAME": VAR_VALUE}` |
|  `project` |  Launch した run を送信する対象の Project |
|  `entity` |  Launch した run を送信する対象の Entity |
|  `queue` |  run を追加するキューの名前 |
|  `priority` |  ジョブの優先度（1 が最も高い優先度） |
|  `resource` |  run の実行バックエンド。W&B では "local-container" バックエンドのサポートがあります。 |
|  `entry_point` |  Project 内で実行するエントリーポイント。wandb URI では元の run の entry point、Git リポジトリの場合は main.py がデフォルトとなります。 |
|  `name` |  Launch する run の名前。 |
|  `version` |  Git ベースの Project の場合はコミットハッシュまたはブランチ名。 |
|  `docker_image` |  run に利用する docker イメージ名。 |
|  `resource_args` |  リモートバックエンドで run を起動するためのリソース関連引数。構成された Launch 設定の `resource_args` として保持されます。 |
|  `run_id` |  Launch された run の ID（オプション） |
|  `build` |  オプションのフラグで、デフォルトは false。build を有効にする場合は queue の指定が必要で、イメージ作成・job artifact の作成・その参照を queue へプッシュします。 |
|  `repository` |  イメージをリモートレジストリへプッシュする際に用いるリモートリポジトリ名を制御する文字列（オプション） |
|  `project_queue` |  キュー用の Project 名を制御する文字列（オプション）。主に Project スコープのキューとの後方互換性用です。 |

#### 例:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B Project をローカルホスト上で再現可能な docker 環境で実行
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
|  `wandb.api.public.QueuedRun` インスタンスを返し、キューイングされた run の情報を取得できます。`wait_until_started` や `wait_until_finished` を呼び出した場合は、run の詳細情報へアクセス可能です。 |

| 例外 |  |
| :--- | :--- |
|  `wandb.exceptions.LaunchError`（失敗時） |