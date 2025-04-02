---
title: init
menu:
  reference:
    identifier: ja-ref-python-init
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_init.py#L1131-L1483 >}}

W&B に track および ログ を記録するための新しい run を開始します。

```python
init(
    entity: (str | None) = None,
    project: (str | None) = None,
    dir: (StrPath | None) = None,
    id: (str | None) = None,
    name: (str | None) = None,
    notes: (str | None) = None,
    tags: (Sequence[str] | None) = None,
    config: (dict[str, Any] | str | None) = None,
    config_exclude_keys: (list[str] | None) = None,
    config_include_keys: (list[str] | None) = None,
    allow_val_change: (bool | None) = None,
    group: (str | None) = None,
    job_type: (str | None) = None,
    mode: (Literal['online', 'offline', 'disabled'] | None) = None,
    force: (bool | None) = None,
    anonymous: (Literal['never', 'allow', 'must'] | None) = None,
    reinit: (bool | None) = None,
    resume: (bool | Literal['allow', 'never', 'must', 'auto'] | None) = None,
    resume_from: (str | None) = None,
    fork_from: (str | None) = None,
    save_code: (bool | None) = None,
    tensorboard: (bool | None) = None,
    sync_tensorboard: (bool | None) = None,
    monitor_gym: (bool | None) = None,
    settings: (Settings | dict[str, Any] | None) = None
) -> Run
```

ML トレーニング パイプラインでは、トレーニング スクリプトの先頭と評価スクリプトに `wandb.init()` を追加できます。各ピースは W&B の run として track されます。

`wandb.init()` は、データを run に記録するための新しいバックグラウンド プロセスを生成し、デフォルトでデータを https://wandb.ai に同期するため、結果をリアルタイムで確認できます。

`wandb.log()` でデータを ログ に記録する前に、`wandb.init()` を呼び出して run を開始します。データの ログ 記録が完了したら、`wandb.finish()` を呼び出して run を終了します。`wandb.finish()` を呼び出さない場合、run はスクリプトの終了時に終了します。

詳細な例を含む `wandb.init()` の使用方法の詳細については、[ガイドと FAQ](https://docs.wandb.ai/guides/track/launch) を参照してください。

#### 例：

### entity と project を明示的に設定し、run の名前を選択します。

```python
import wandb

run = wandb.init(
    entity="geoff",
    project="capsules",
    name="experiment-2021-10-31",
)

# ... ここにトレーニングコードを記述 ...

run.finish()
```

### `config` 引数を使用して、run に関する メタデータ を追加します。

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    run.config.update({"architecture": "resnet", "depth": 34})

    # ... ここにトレーニングコードを記述 ...
```

`wandb.init()` をコンテキスト マネージャーとして使用して、ブロックの最後に `wandb.finish()` を自動的に呼び出すことができることに注意してください。

| Args |  |
| :--- | :--- |
| `entity` | run が ログ される ユーザー名または Team 名。 entity は既に存在している必要があります。run の ログ 記録を開始する前に、UI でアカウントまたは Team を作成していることを確認してください。指定されていない場合、run はデフォルトの entity にデフォルト設定されます。デフォルトの entity を変更するには、[設定](https://wandb.ai/settings) に移動し、[デフォルトの Team] の下の [新しい project を作成するデフォルトの場所] を更新します。 |
| `project` | この run が ログ される project の名前。指定されていない場合、git root や現在のプログラム ファイルの確認など、システムに基づいて project 名を推測するヒューリスティックを使用します。project 名を推測できない場合、project は `"uncategorized"` にデフォルト設定されます。 |
| `dir` | 実験 ログ と メタデータ ファイルが保存されるディレクトリーへの絶対パス。指定されていない場合、これはデフォルトで `./wandb` ディレクトリーになります。これは、`download()` を呼び出すときに Artifacts が保存される場所に影響しないことに注意してください。 |
| `id` | 再開に使用される、この run の一意の識別子。project 内で一意である必要があり、run が削除されると再利用できません。識別子に、次の特殊文字を含めることはできません：`/ \ # ? % :`。短い説明的な名前には、`name` フィールドを使用します。run 間で比較するために ハイパーパラメーター を保存するには、`config` を使用します。 |
| `name` | この run の短い表示名。UI に表示され、run を識別するのに役立ちます。デフォルトでは、テーブルからチャートへの簡単な相互参照 run を可能にする、ランダムな 2 語の名前を生成します。これらの run 名を短く保つと、チャートの凡例とテーブルの可読性が向上します。ハイパーパラメーター を保存するには、`config` フィールドを使用することをお勧めします。 |
| `notes` | Git のコミット メッセージと同様に、run の詳細な説明。この引数を使用して、今後この run の目的または設定を思い出すのに役立つコンテキストまたは詳細をキャプチャします。 |
| `tags` | UI でこの run にラベルを付けるためのタグのリスト。タグは、run の整理や、"ベースライン" や "プロダクション" などの一時的な識別子の追加に役立ちます。UI でタグを簡単に追加、削除、またはタグでフィルタリングできます。run を再開する場合、ここで指定されたタグは、既存のタグを置き換えます。現在のタグを上書きせずに、再開された run にタグを追加するには、`run = wandb.init()` を呼び出した後で、`run.tags += ["new_tag"]` を使用します。 |
| `config` | `wandb.config` を設定します。これは、モデルの ハイパーパラメーター や データ の 前処理 設定など、run への入力 パラメータ を保存するための辞書のようなオブジェクトです。config は UI の概要ページに表示され、これらの パラメータ に基づいて run をグループ化、フィルタリング、および並べ替えることができます。キー にはピリオド（`.`）を含めないでください。また、値は 10 MB 未満にする必要があります。辞書、`argparse.Namespace`、または `absl.flags.FLAGS` が指定されている場合、キーと値のペア は `wandb.config` に直接ロードされます。文字列が指定されている場合は、YAML ファイルへのパスとして解釈され、そこから構成値が `wandb.config` にロードされます。 |
| `config_exclude_keys` | `wandb.config` から除外する特定のキーのリスト。 |
| `config_include_keys` | `wandb.config` に含める特定のキーのリスト。 |
| `allow_val_change` | config 値を最初に設定した後で変更できるかどうかを制御します。デフォルトでは、config 値が上書きされた場合、例外が発生します。学習率など、トレーニング 中に変化する変数を track する場合は、代わりに `wandb.log()` を使用することを検討してください。デフォルトでは、これはスクリプトでは `False` で、Notebook 環境では `True` です。 |
| `group` | より大規模な実験の一部として個々の run を整理するために、グループ名を指定します。これは、交差検証や、異なる テストセット でモデルをトレーニングおよび評価する複数のジョブを実行する場合に役立ちます。グループ化を使用すると、UI で関連する run をまとめて管理できるため、統一された実験として結果を簡単に切り替えて確認できます。詳細については、[run のグループ化に関するガイド](https://docs.wandb.com/guides/runs/grouping) を参照してください。 |
| `job_type` | 特に、より大規模な実験の一部としてグループ内の run を整理する場合に役立つ、run のタイプを指定します。たとえば、グループでは、"train" や "eval" などの ジョブタイプ で run にラベルを付けることができます。ジョブタイプ を定義すると、UI で同様の run を簡単にフィルタリングおよびグループ化できるため、直接比較が容易になります。 |
| `mode` | run データの管理方法を指定します。次のオプションがあります。- `"online"`（デフォルト）：ネットワーク接続が利用可能な場合、W&B とのライブ同期を有効にし、可視化をリアルタイムで更新します。- `"offline"`：エアギャップ環境またはオフライン環境に適しています。データはローカルに保存され、後で同期できます。将来の同期を有効にするには、run フォルダーが保持されていることを確認してください。- `"disabled"`：すべての W&B 機能を無効にし、run の メソッド を no-op にします。通常、W&B 操作をバイパスするためにテストで使用されます。 |
| `force` | スクリプトを実行するために W&B ログインが必要かどうかを決定します。`True` の場合、ユーザー は W&B にログインしている必要があります。そうでない場合、スクリプトは続行されません。`False` （デフォルト）の場合、ユーザー がログインしていなくてもスクリプトは続行でき、ユーザー がログインしていない場合はオフライン モードに切り替わります。 |
| `anonymous` | 匿名 データ の ログ 記録に対する制御のレベルを指定します。使用可能なオプションは次のとおりです。- `"never"`（デフォルト）：run を track する前に、W&B アカウントをリンクする必要があります。これにより、各 run がアカウントに関連付けられるようにすることで、意図しない匿名の run の作成を防ぎます。- `"allow"`：ログイン ユーザー がアカウントで run を track できるようにするだけでなく、W&B アカウントなしでスクリプトを実行している ユーザー が UI でチャートと データを表示できるようにします。- `"must"`：ユーザー がログインしている場合でも、run が匿名アカウントに ログ されるように強制します。 |
| `reinit` | 複数の `wandb.init()` 呼び出しが、同じ プロセス 内で新しい run を開始できるかどうかを決定します。デフォルト（`False`）では、アクティブな run が存在する場合、`wandb.init()` を呼び出すと、新しい run を作成する代わりに、既存の run が返されます。`reinit=True` の場合、新しい run が初期化される前に、アクティブな run が終了します。Notebook 環境では、`reinit` が明示的に `False` に設定されていない限り、run はデフォルトで再初期化されます。 |
| `resume` | 指定された `id` で run を再開する際の 振る舞い を制御します。使用可能なオプションは次のとおりです。- `"allow"`：指定された `id` を持つ run が存在する場合、最後のステップから再開されます。それ以外の場合は、新しい run が作成されます。- `"never"`：指定された `id` を持つ run が存在する場合、エラーが発生します。そのような run が見つからない場合は、新しい run が作成されます。- `"must"`：指定された `id` を持つ run が存在する場合、最後のステップから再開されます。run が見つからない場合は、エラーが発生します。- `"auto"`：このマシンでクラッシュした場合、以前の run を自動的に再開します。それ以外の場合は、新しい run を開始します。- `True`：非推奨。代わりに `"auto"` を使用してください。- `False`：非推奨。常に新しい run を開始するには、デフォルトの 振る舞い （`resume` を設定しないままにする）を使用してください。注：`resume` が設定されている場合、`fork_from` と `resume_from` は使用できません。`resume` が設定されていない場合、システムは常に新しい run を開始します。詳細については、[run の再開に関するガイド](https://docs.wandb.com/guides/runs/resuming) を参照してください。 |
| `resume_from` | 形式 `{run_id}?_step={step}` を使用して、以前の run の特定の時点から run を再開することを指定します。これにより、ユーザー は run に ログ された履歴を途中のステップで切り捨て、そのステップから ログ 記録を再開できます。ターゲット run は、同じ project 内にある必要があります。`id` 引数も指定されている場合、`resume_from` 引数が優先されます。`resume`、`resume_from`、および `fork_from` は一緒に使用できず、一度に 1 つだけ使用できます。注：この機能はベータ版であり、将来変更される可能性があります。 |
| `fork_from` | 形式 `{id}?_step={step}` を使用して、以前の run の特定の時点から新しい run をフォークすることを指定します。これにより、ターゲット run の履歴の指定されたステップから ログ 記録を再開する新しい run が作成されます。ターゲット run は、現在の project の一部である必要があります。`id` 引数も指定されている場合、`fork_from` 引数とは異なる必要があります。同じ場合は、エラーが発生します。`resume`、`resume_from`、および `fork_from` は一緒に使用できず、一度に 1 つだけ使用できます。注：この機能はベータ版であり、将来変更される可能性があります。 |
| `save_code` | メイン スクリプト または Notebook を W&B に保存できるようにします。これにより、実験 の 再現性 が向上し、UI で run 間の コード 比較が可能になります。デフォルトでは、これは無効になっていますが、[設定ページ](https://wandb.ai/settings) でデフォルトを変更して有効にすることができます。 |
| `tensorboard` | 非推奨。代わりに `sync_tensorboard` を使用してください。 |
| `sync_tensorboard` | TensorBoard または TensorBoardX から W&B ログ の自動同期を有効にし、W&B UI で表示するための関連 イベント ファイルを保存します。関連 イベント ファイルを保存して、W&B UI で表示します。（デフォルト：`False`） |
| `monitor_gym` | OpenAI Gym を使用する場合に、環境のビデオ の自動 ログ 記録を有効にします。詳細については、[gym インテグレーション に関するガイド](https://docs.wandb.com/guides/integrations/openai-gym) を参照してください。 |
| `settings` | run の詳細設定を含む辞書または `wandb.Settings` オブジェクトを指定します。 |

| Returns |  |
| :--- | :--- |
| 現在の run へのハンドルである `Run` オブジェクト。このオブジェクトを使用して、データの ログ 記録、ファイルの保存、run の終了などの操作を実行します。詳細については、[Run API](https://docs.wandb.ai/ref/python/run) を参照してください。 |

| Raises |  |
| :--- | :--- |
| `Error` | run の初期化中に不明なエラーまたは内部エラーが発生した場合。 |
| `AuthenticationError` | ユーザー が有効な認証情報の提供に失敗した場合。 |
| `CommError` | W&B サーバー との通信に問題が発生した場合。 |
| `UsageError` | ユーザー が関数に無効な 引数 を指定した場合。 |
| `KeyboardInterrupt` | ユーザー が run の初期化 プロセス を中断した場合。ユーザー が run の初期化 プロセス を中断した場合。 |
