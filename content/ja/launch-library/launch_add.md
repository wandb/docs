---
title: Launch を追加
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch_add.py#L34-L131 >}}

W&B Launch の 実験 を キューに入れます。ソース URI、job、または docker_image の いずれかを 指定します。

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
|  `uri` |  実行する 実験 の URI。wandb の run URI または Git リポジトリ の URI。 |
|  `job` |  wandb.Job への 文字列参照。例: wandb/test/my-job:latest |
|  `config` |  run の 設定 を 含む 辞書。キー "resource_args" の 下に リソース 固有 の 引数 を 含める ことも できます。 |
|  `template_variables` |  run キュー 用 の テンプレート 変数 の 値 を 含む 辞書。期待 される 形式 は `{"VAR_NAME": VAR_VALUE}` です。 |
|  `project` |  起動 した run の 送信先 Project。 |
|  `entity` |  起動 した run の 送信先 Entity。 |
|  `queue` |  run を キューに 入れる 先 の キュー 名。 |
|  `priority` |  job の 優先度 レベル。1 が 最高 優先度 です。 |
|  `resource` |  run の 実行 バックエンド。W&B は "local-container" バックエンド を 標準 サポート します。 |
|  `entry_point` |  Project 内で 実行 する エントリーポイント。wandb の URI の 場合 は 元 の run で 使用 された エントリーポイント、Git リポジトリ の URI の 場合 は main.py が 使われます。 |
|  `name` |  起動 する run に 付与 する 名前。 |
|  `version` |  Git ベース の Project の 場合、コミット ハッシュ または ブランチ 名。 |
|  `docker_image` |  run で 使用 する Docker イメージ 名。 |
|  `resource_args` |  リモート バックエンド で run を 起動 する 際 の リソース 関連 引数。作成 された Launch 設定 の `resource_args` に 保存 されます。 |
|  `run_id` |  起動 した run の ID を 示す 任意 の 文字列。 |
|  `build` |  任意 の フラグ で 既定 は false。build が 有効 な 場合 は キュー の 設定 が 必要 で、イメージ が 作成 され、job の Artifact が 作成 され、その Artifact への 参照 が キュー に プッシュ されます。 |
|  `repository` |  リモート リポジトリ 名 を 制御 する 任意 の 文字列。レジストリ に イメージ を プッシュ する 際 に 使用 されます。 |
|  `project_queue` |  キュー の Project 名 を 制御 する 任意 の 文字列。主に Project スコープ の キュー との 後方 互換性 の ため に 使用 されます。 |

#### 例:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B Project を 実行 し、再現 可能 な Docker 環境 を 作成 します
# ローカル ホスト 上で
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
|  キュー に 入れられた run に 関する 情報 を 返す `wandb.api.public.QueuedRun` の インスタンス。`wait_until_started` または `wait_until_finished` が 呼ばれた 場合 は、背後 の Run 情報 へ の アクセス を 提供 します。 |

| 例外 |  |
| :--- | :--- |
|  失敗 した 場合 は `wandb.exceptions.LaunchError` を 送出 します。 |