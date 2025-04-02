---
title: launch_add
menu:
  reference:
    identifier: ja-ref-python-launch-library-launch_add
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/_launch_add.py#L34-L131 >}}

W&B の Launch の experiment をエンキューします。ソース URI 、job 、または docker_image のいずれかを使用します。

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
| `uri` | 実行する experiment の URI 。Wandb の run の URI または Git リポジトリの URI 。 |
| `job` | wandb.Job への文字列参照（例：wandb/test/my-job:latest） |
| `config` | run の 設定 を含む 辞書 。キー "resource_args" の下にあるリソース固有の 引数 も含めることができます。 |
| `template_variables` | run queue の テンプレート 変数の 値 を含む 辞書 。`{"VAR_NAME": VAR_VALUE}` の 形式 が想定されます。 |
| `project` | Launch された run の送信先となるターゲット プロジェクト |
| `entity` | Launch された run の送信先となるターゲット エンティティ |
| `queue` | run をエンキューする queue の 名前 |
| `priority` | job の 優先度 。1 が 最も高い 優先度 です。 |
| `resource` | run の 実行 バックエンド：W&B は "local-container" バックエンドの組み込みサポートを提供します。 |
| `entry_point` | プロジェクト 内で 実行 する エントリーポイント。wandb URI の 場合は 元の run で 使用 された エントリーポイント、Git リポジトリ URI の 場合は main.py を デフォルト で 使用 します。 |
| `name` | run を Launch する際に 使用 する run の 名前 。 |
| `version` | Git ベース の プロジェクト の 場合、コミット ハッシュ または ブランチ 名 のいずれか。 |
| `docker_image` | run に 使用 する docker イメージ の 名前 。 |
| `resource_args` | リモート バックエンド への run の Launch に 関連 する リソース 引数 。`resource_args` の下で 構築 された Launch 設定 に 保存 されます。 |
| `run_id` | Launch された run の ID を 示す オプション の 文字列 |
| `build` | オプション の フラグ で、デフォルト は False です。queue を 設定 する 必要 が ある 場合、イメージ が 作成 され、job アーティファクト が 作成 され、その job アーティファクト への 参照 が queue に プッシュ されます。 |
| `repository` | イメージ を レジストリ に プッシュ する 際 に 使用 される、リモート リポジトリ の 名前 を 制御 する オプション の 文字列。 |
| `project_queue` | queue の プロジェクト の 名前 を 制御 する オプション の 文字列。主に プロジェクト スコープ の queue との 後方 互換性 のために 使用 されます。 |

#### 例：

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# Run W&B プロジェクト と 再現 可能 な docker 環境 を 作成 します
# ローカル ホスト 上 で
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 戻り値 |  |
| :--- | :--- |
| `wandb.api.public.QueuedRun` の インスタンス。キューに 入った run に関する 情報 を 提供 し、`wait_until_started` または `wait_until_finished` が 呼び出さ れた 場合 は、基盤 と なる Run 情報 への アクセス を 提供 します。 |

| 例外 |  |
| :--- | :--- |
| 失敗 した 場合 の `wandb.exceptions.LaunchError` |
