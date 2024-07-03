# init

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_init.py#L924-L1186' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

新しい run を開始し、W&B でトラッキングとログを行います。

```python
init(
    job_type: Optional[str] = None,
    dir: Optional[StrPath] = None,
    config: Union[Dict, str, None] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    reinit: Optional[bool] = None,
    tags: Optional[Sequence] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    magic: Optional[Union[dict, str, bool]] = None,
    config_exclude_keys: Optional[List[str]] = None,
    config_include_keys: Optional[List[str]] = None,
    anonymous: Optional[str] = None,
    mode: Optional[str] = None,
    allow_val_change: Optional[bool] = None,
    resume: Optional[Union[bool, str]] = None,
    force: Optional[bool] = None,
    tensorboard: Optional[bool] = None,
    sync_tensorboard: Optional[bool] = None,
    monitor_gym: Optional[bool] = None,
    save_code: Optional[bool] = None,
    id: Optional[str] = None,
    fork_from: Optional[str] = None,
    resume_from: Optional[str] = None,
    settings: Union[Settings, Dict[str, Any], None] = None
) -> Union[Run, RunDisabled]
```

MLトレーニングパイプラインでは、トレーニングスクリプトおよび評価スクリプトの最初に `wandb.init()` を追加し、各部分を W&B で run としてトラッキングすることができます。

`wandb.init()` はバックグラウンドプロセスを新しく起動して run にデータをログし、デフォルトで wandb.ai にデータを同期します。これによりリアルタイムの可視化が可能です。

`wandb.log()` でデータをログする前に、`wandb.init()` を呼び出して run を開始します。

```python
import wandb

wandb.init()
# ... メトリクスの計算やメディアの生成
wandb.log({"accuracy": 0.9})
```

`wandb.init()` は run オブジェクトを返し、`wandb.run` からもアクセス可能です。

```python
import wandb

run = wandb.init()

assert run is wandb.run
```

スクリプトの最後で `wandb.finish` を自動的に呼び出して、run の終了とクリーンアップを行います。ただし、子プロセスから `wandb.init` を呼び出す場合は、子プロセスの終了時に明示的に `wandb.finish` を呼び出す必要があります。

`wandb.init()` の使用方法の詳細な例については、[ガイドとFAQ](https://docs.wandb.ai/guides/track/launch)をご覧ください。

| 引数 |  |
| :--- | :--- |
|  `project` |  (str, optional) 新しい run を送信するプロジェクトの名前。プロジェクトが指定されない場合、run は "Uncategorized" プロジェクトに配置されます。 |
|  `entity` |  (str, optional) run を送信するユーザー名またはチーム名。この entity は run を送信する前に存在している必要があるため、UIでアカウントまたはチームを作成してください。entity を指定しない場合、run はデフォルト entity に送信されます。デフォルト entity は [設定](https://wandb.ai/settings) で変更できます。 |
|  `config` |  (dict, argparse, absl.flags, str, optional) `wandb.config` を設定するための辞書形式のオブジェクト。モデルのハイパーパラメータやデータ前処理ジョブの設定などを保存します。この config は UI のテーブルに表示され、グループ化、フィルタリング、ソートが可能です。キーには `.` を含めず、値は 10 MB 以下である必要があります。dict, argparse, absl.flags の場合: `wandb.config` オブジェクトにキーバリューペアをロードします。str の場合: その名前の yaml ファイルを探し、そのファイルから config をロードします。 |
|  `save_code` |  (bool, optional) メインスクリプトまたはノートブックを W&B に保存します。実験の再現性を向上させ、UIで実験間のコードの差分を確認するのに役立ちます。デフォルトではオフですが、[設定ページ](https://wandb.ai/settings) でデフォルトの挙動をオンに変更できます。 |
|  `group` |  (str, optional) 個別の run を大規模な実験に整理するためのグループを指定します。例えば、交差検証を行う場合や、異なるテストセットに対してモデルをトレーニングおよび評価する複数のジョブがある場合などです。詳しくは [run のグループ化に関するガイド](https://docs.wandb.com/guides/runs/grouping) を参照してください。 |
|  `job_type` |  (str, optional) run のタイプを指定します。例えば、train と eval など、グループ内の類似した run をフィルタリングおよびグループ化するのに役立ちます。 |
|  `tags` |  (list, optional) この run のタグのリスト。タグは run を整理するのに役立ちます。`wandb.init()` に渡したタグによって run のタグが書き換えられます。再開した run にタグを追加したい場合は、 `wandb.init()` の後に `run.tags += ["new_tag"]` を使用します。 |
|  `name` |  (str, optional) UI でこの run を識別するための短い表示名。デフォルトではランダムな2語の名前が生成され、表とチャートで簡単にクロスリファレンスできます。 |
|  `notes` |  (str, optional) run の長い説明。例: `-m` コミットメッセージのようなもの。 |
|  `dir` |  (str または pathlib.Path, optional) メタデータが保存されるディレクトリへの絶対パス。デフォルトは `./wandb` ディレクトリです。 |
|  `resume` |  (bool, str, optional) 再開の挙動を設定します。オプション: `"allow"`, `"must"`, `"never"`, `"auto"` または `None`。デフォルトは `None`。詳しくは [run の再開に関するガイド](https://docs.wandb.com/guides/runs/resuming) を参照してください。 |
|  `reinit` |  (bool, optional) 同じプロセス内で複数の `wandb.init()` 呼び出しを許可します。デフォルトは `False`。 |
|  `magic` |  (bool, dict, または str, optional) スクリプトを自動インストルメントするかどうかを設定します。デフォルトは `False`。辞書、json 文字列、または yaml ファイル名も渡すことができます。 |
|  `config_exclude_keys` |  (list, optional) `wandb.config` から除外するキーのリスト。 |
|  `config_include_keys` |  (list, optional) `wandb.config` に含めるキーのリスト。 |
|  `anonymous` |  (str, optional) 匿名データのログを制御します。オプション: - `"never"` (デフォルト): run をトラッキングする前に W&B アカウントにリンクする必要があります。 - `"allow"`: ログインしているユーザーは自分のアカウントで run をトラッキングできますが、W&B アカウントを持っていないユーザーも UI でチャートを見ることができます。 - `"must"`: run を匿名アカウントに送信します。 |
|  `mode` |  (str, optional) `"online"`, `"offline"` または `"disabled"` に設定できます。デフォルトは online です。 |
|  `allow_val_change` |  (bool, optional) 一度設定されたキーの後に config の値を変更できるかどうか。デフォルトは、config 値が上書きされると例外をスローします。 (スクリプトではデフォルトで `False`、Jupyter では `True`) |
|  `force` |  (bool, optional) `True` の場合、W&B にログインしていない場合にスクリプトがクラッシュします。`False` の場合、W&B にログインしていない場合はオフラインモードでスクリプトを実行します。デフォルトは `False`。 |
|  `sync_tensorboard` |  (bool, optional) tensorboard または tensorboardX から wandb のログを同期し、関連するイベントファイルを保存します。デフォルトは `False` です。 |
|  `monitor_gym` |  (bool, optional) OpenAI Gym 環境で動画を自動的にログします。デフォルトは `False`。詳しくは [このインテグレーションに関するガイド](https://docs.wandb.com/guides/integrations/openai-gym) を参照してください。 |
|  `id` |  (str, optional) 再開に使用するこの run の一意の ID。プロジェクト内で一意である必要があり、run を削除した場合は再利用できません。 ID に `/\#?%:` などの特殊文字を含めることはできません。詳しくは [run の再開に関するガイド](https://docs.wandb.com/guides/runs/resuming) を参照してください。 |
|  `fork_from` |  (str, optional) {run_id}?_step={step} の形式で、以前の run から新しい run をフォークする瞬間を記述した文字列。この形式で新しい run を作成し、指定された瞬間に前の run のログ履歴を引き継ぎます。対象の run は現在のプロジェクトに存在している必要があります。例: `fork_from="my-run-id?_step=1234"`。 |

#### 例:

### run のログ場所を設定

run のログ場所を変更することができます。Git の組織、リポジトリ、およびブランチを変更するのと同じように:

```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### run に関するメタデータをconfigに追加

ハイパーパラメータなどのメタデータを run に追加するために、`config` キーワード引数として辞書形式のオブジェクトを渡します。

```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| 例外 |  |
| :--- | :--- |
|  `Error` |  run の初期化中に未知のエラーまたは内部エラーが発生した場合。 |
|  `AuthenticationError` |  ユーザーが有効な認証情報を提供できなかった場合。 |
|  `CommError` |  WandB サーバーとの通信に問題があった場合。 |
|  `UsageError` |  ユーザーが無効な引数を提供した場合。 |
|  `KeyboardInterrupt` |  ユーザーが run を中断した場合。 |

| 戻り値 |  |
| :--- | :--- |
|  `Run` オブジェクト。 |