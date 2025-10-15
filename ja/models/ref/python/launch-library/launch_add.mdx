---
title: 'launch_add


  '
menu:
  reference:
    identifier: ja-ref-python-launch-library-launch_add
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/_launch_add.py#L34-L131 >}}

W&B ローンチ実験をキューに追加します。ソース URI、ジョブ、または docker_image のいずれかを使用します。

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
|  `uri` |  実行する実験の URI。wandb の run URI や Git リポジトリ URI。 |
|  `job` |  wandb.Job への文字列参照。例: wandb/test/my-job:latest |
|  `config` |  run の設定を含む辞書。「resource_args」キーの下に、リソース特有の引数も含めることができます。 |
|  `template_variables` |  キューのテンプレート変数の値を含む辞書。`{"VAR_NAME": VAR_VALUE}` の形式を期待します。 |
|  `project` |  起動する run を送信する対象のプロジェクト |
|  `entity` |  起動する run を送信する対象のエンティティ |
|  `queue` |  run をキューに追加するためのキューの名前 |
|  `priority` |  ジョブの優先度レベルで、1 が最高の優先度 |
|  `resource` |  run の実行バックエンド: W&B は「local-container」バックエンドを標準サポート |
|  `entry_point` |  プロジェクト内で実行するエントリーポイント。デフォルトでは、wandb URI の場合はオリジナルの run で使用されたエントリーポイントを、Git リポジトリ URI の場合は main.py を使用します。 |
|  `name` |  run を起動する際の名前。 |
|  `version` |  Git ベースのプロジェクトの場合、コミットハッシュまたはブランチ名。 |
|  `docker_image` |  run に使用する Docker イメージの名前。 |
|  `resource_args` |  リモートバックエンドに run を起動するためのリソース関連の引数。構築されたローンチ設定の `resource_args` に保存されます。 |
|  `run_id` |  起動された run の ID を示すオプションの文字列 |
|  `build` |  デフォルトで false のオプションフラグ。queue がセットされている場合に必要です。イメージを作成し、ジョブアーティファクトを作成し、そのジョブアーティファクトへの参照をキューにプッシュします。 |
|  `repository` |  画像をレジストリにプッシュする際に使用するリモートリポジトリの名前を制御するオプションの文字列 |
|  `project_queue` |  キュー用のプロジェクト名を制御するためのオプションの文字列。主にプロジェクトにスコープされたキューとの後方互換性のために使用されます。 |

#### 例:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B プロジェクトを実行し、ローカルホスト上で再現可能な Docker 環境を作成
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
| `wandb.api.public.QueuedRun` のインスタンスで、キューに追加された run に関する情報を提供します。また、`wait_until_started` または `wait_until_finished` が呼び出された場合、基礎となる Run 情報にアクセスできます。 |

| 例外 |  |
| :--- | :--- |
| `wandb.exceptions.LaunchError` が失敗時に発生します |
