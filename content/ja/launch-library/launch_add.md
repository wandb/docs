---
title: launch_add
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch_add.py#L34-L131 >}}

W&B Launch 実験をキューに追加します。source uri、job、docker_image のいずれかを指定できます。

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

| 引数 | 説明 |
| :--- | :--- |
|  `uri` |  実行する実験の URI。wandb run の uri または Git リポジトリの URI を指定します。 |
|  `job` |  wandb.Job への参照文字列（例: wandb/test/my-job:latest） |
|  `config` |  run に対する設定情報を格納した辞書。`resource_args` キーの下にリソース特有の引数も含められます。 |
|  `template_variables` |  run キュー用のテンプレート変数の値を持つ辞書。期待される形式は `{"VAR_NAME": VAR_VALUE}` です。 |
|  `project` |  実行した run を送信する対象の Project |
|  `entity` |  実行した run を送信する対象の Entity |
|  `queue` |  run をキューに追加する際のキュー名 |
|  `priority` |  ジョブの優先度（1 が最高） |
|  `resource` |  run の実行バックエンド（W&B では "local-container" バックエンドに標準対応しています） |
|  `entry_point` |  Project 内で実行するエントリーポイント。wandb URI の場合は元の run で使われていた entry point、Git リポジトリの場合は main.py がデフォルト。 |
|  `name` |  run の名前。 |
|  `version` |  Git ベースのプロジェクト用。コミットハッシュまたはブランチ名。 |
|  `docker_image` |  run で使用する docker イメージ名。 |
|  `resource_args` |  リモートバックエンドへの run 実行時に使用するリソース関連の引数。`resource_args` の形で Launch 設定に保存されます。 |
|  `run_id` |  Launch した run の ID を指定（オプション） |
|  `build` |  デフォルトは False。True の場合、queue の設定が必要で、イメージを作成しジョブ Artifacts を生成し、その参照を queue へ送信します。 |
|  `repository` |  オプション。リモートリポジトリ名の制御に使用します。イメージをレジストリに push する際に利用。 |
|  `project_queue` |  オプション。キュー用プロジェクト名の制御に使用。主に Project 単位のキューとの互換性を保持するために利用。 |

#### 例:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B Project をローカルで実行し、再現性のある docker 環境を構築
# ローカルホスト上で実行します
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 戻り値 | 説明 |
| :--- | :--- |
|  `wandb.api.public.QueuedRun` のインスタンス。キュー追加された run の情報を取得可能。また、`wait_until_started` や `wait_until_finished` を呼び出すことで、対応する Run の詳細情報へアクセスできます。 |

| 例外 | 説明 |
| :--- | :--- |
|  `wandb.exceptions.LaunchError` が発生した場合、処理に失敗したことを示します。 |