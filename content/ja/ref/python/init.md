---
title: init
menu:
  reference:
    identifier: ja-ref-python-init
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_init.py#L1131-L1483 >}}

新規のRunを開始して、W&Bにトラックしてログします。

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

MLトレーニングパイプラインでは、トレーニングスクリプトや評価スクリプトの最初に`wandb.init()`を追加することができます。それぞれの部分はW&BでのRunとしてトラックされます。

`wandb.init()`はバックグラウンドプロセスを立ち上げてRunにデータをログし、デフォルトで https://wandb.ai と同期して、リアルタイムで結果を見ることができます。

データを`wandb.log()`でログする前に、Runを開始するために`wandb.init()`を呼び出します。データのログが終わったら、`wandb.finish()`を呼び出してRunを終了します。`wandb.finish()`を呼び出さない場合は、スクリプトが終了した時にRunが終了します。

`wandb.init()`の使い方や詳細な例については、[ガイドとFAQ](https://docs.wandb.ai/guides/track/launch)をご覧ください。

#### 例:

### entityとプロジェクトを明示的に設定し、Runの名前を選択します:

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

### `config`引数を使ってRunに関するメタデータを追加します:

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    run.config.update({"architecture": "resnet", "depth": 34})

    # ... ここにトレーニングコードを記述 ...
```

`wandb.init()`をコンテキストマネージャとして使用することで、ブロックの終了時に自動的に`wandb.finish()`を呼び出すことができます。

| 引数 |  |
| :--- | :--- |
|  `entity` |  Runがログされるユーザー名またはチーム名。entityはすでに存在している必要があるため、Runをログし始める前にUIでアカウントまたはチームを作成したことを確認してください。指定しない場合、Runはデフォルトのエンティティにデフォルトされます。デフォルトのentityを変更するには、[あなたの設定](https://wandb.ai/settings)にアクセスして、「新規プロジェクトを作成するデフォルトの場所」を「デフォルトチーム」の下で更新します。 |
|  `project` |  このRunがログされるプロジェクトの名前。指定しない場合、プロジェクト名はシステムに基づいて推測され、gitルートや現在のプログラムファイルをチェックします。プロジェクト名を推測できない場合、プロジェクトはデフォルトで`"uncategorized"`になります。 |
|  `dir` |  実験ログとメタデータファイルが保存されるディレクトリへの絶対パス。指定しない場合、デフォルトで`./wandb`ディレクトリになります。`download()`を呼び出したときにアーティファクトが保存される場所には影響しませんので注意してください。 |
|  `id` |  このRunの一意の識別子で、再開に使用されます。プロジェクト内で一意である必要があり、Runが削除された場合は再利用できません。この識別子には次の特殊文字を含むことはできません: `/ \ # ? % :`。短い記述的な名前には`name`フィールドを使用するか、Run間で比較するハイパーパラメータを保存する場合は`config`を使用します。 |
|  `name` |  このRunのUIに表示される短い表示名で、識別を助けます。デフォルトでは、テーブルからチャートへの容易なクロスリファレンスを可能にするランダムな二語の名前が生成されます。これらのRun名を短く保つことで、チャートの凡例やテーブルの可読性が向上します。ハイパーパラメータを保存するには、`config`フィールドを使用することをお勧めします。 |
|  `notes` |  Runの詳細な説明で、Gitのコミットメッセージに似ています。この引数を使用して、将来的にRunの目的やセットアップを思い出すのに役立つ文脈や詳細を記録します。 |
|  `tags` |  UIでこのRunにラベルを付けるためのタグのリスト。タグはRunを整理したり、"baseline"や"production"のような一時的な識別子を追加するのに役立ちます。UIでタグを簡単に追加、削除したり、タグでフィルタリングすることができます。Runを再開する場合、ここで提供されたタグは既存のタグを置き換えます。現在のタグを上書きせずに再開したRunにタグを追加するには、`run = wandb.init()`を呼び出した後に`run.tags += ["new_tag"]`を使用します。 |
|  `config` |  `wandb.config`を設定し、Runへの入力パラメータ、例えばモデルハイパーパラメータやデータ前処理の設定を保存するための辞書のようなオブジェクトです。configはUIの概要ページに表示され、これらのパラメータに基づいてRunをグループ化、フィルタ、並べ替えることができます。キーにはピリオド（`.`）を含めることができず、値は10 MBより小さくする必要があります。辞書、`argparse.Namespace`、または`absl.flags.FLAGS`が提供された場合、キーと値のペアは直接`wandb.config`にロードされます。文字列が提供された場合、それはYAMLファイルへのパスと解釈され、設定値が`wandb.config`にロードされます。 |
|  `config_exclude_keys` |  `wandb.config`から除外する特定のキーのリスト。 |
|  `config_include_keys` |  `wandb.config`に含める特定のキーのリスト。 |
|  `allow_val_change` |  configの値を最初に設定した後に変更可能かどうかを制御します。デフォルトでは、config値が上書きされた場合に例外が発生します。変化する変数をトレーニング中にトラックするには、`wandb.log()`を使用することをお勧めします。デフォルトでは、スクリプトでは`False`で、ノートブック環境では`True`です。 |
|  `group` |  より大きな実験の一部として個別のRunを整理するためのグループ名を指定します。これは、クロスバリデーションや異なるテストセットでモデルをトレーニングして評価する複数のジョブを実行する場合のようなケースで便利です。グループ化を使用することで、関連するRunをUI上でまとめて管理し、一体化した実験として結果を簡単に切り替えたりレビューできます。詳細については、[Runのグループ化に関するガイド](https://docs.wandb.com/guides/runs/grouping)を参照してください。 |
|  `job_type` |  特にグループでのRunを整理する際に役立ちます。例えば、グループ内では、ジョブタイプとして"train"や"eval"などのラベルをRunに付けることができます。ジョブタイプを定義することで、UIで類似のRunを簡単にフィルタ、グループ化し、直接的な比較を促進します。 |
|  `mode` |  Runデータの管理方法を指定します。選択可能なオプションは次のとおりです： - `"online"`（デフォルト）: ネットワーク接続が利用可能な場合、W&Bとライブ同期が可能で、可視化がリアルタイムで更新されます。 - `"offline"`: エアギャップ環境やオフライン環境に適しており、データはローカルに保存され、後で同期できます。後の同期を可能にするためには、Runフォルダーの保存を保証してください。 - `"disabled"`: すべてのW&B機能を無効にし、Runメソッドをno-opにします。通常、W&B操作をバイパスするためのテストに使用されます。 |
|  `force` |  スクリプトの実行にW&Bへのログインが必要かどうかを決定します。`True`の場合、ユーザーはW&Bにログインしていなければスクリプトを実行できません。`False`（デフォルト）の場合、ユーザーがログインしていない場合でもスクリプトはオフラインモードで続行することができます。 |
|  `anonymous` |  匿名データのログレベルを指定します。利用可能なオプションは次のとおりです： - `"never"`（デフォルト）: Runをトラックする前にW&Bアカウントをリンクする必要があります。これにより、各Runがアカウントに関連付けられていることを保証し、匿名Runの偶発的な作成を防ぎます。 - `"allow"`: ログイン済みのユーザーがアカウントでRunをトラックすることができますが、W&Bアカウントを持っていない人がスクリプトを実行してもUIでチャートとデータを表示できます。 - `"must"`: ユーザーがログインしていても、Runが匿名のアカウントにログされることを強制します。 |
|  `reinit` |  同一プロセス内で複数の`wandb.init()`呼び出しが新規のRunを開始できるかどうかを決定します。デフォルト（`False`）では、アクティブなRunが存在する場合、`wandb.init()`を呼び出すと新しいRunを作成せずに既存のRunを返します。`reinit=True`の場合、アクティブなRunは新しいRunが初期化される前に終了されます。ノートブック環境では、`reinit`が`False`に設定されていない限り、Runはデフォルトで再初期化されます。 |
|  `resume` |  指定された`id`を使用してRunの再開時の動作を制御します。利用可能なオプションは次のとおりです： - `"allow"`: 指定された`id`のRunが存在する場合、最後のステップから再開し、そうでない場合は新しいRunが作成されます。 - `"never"`: 指定された`id`のRunが存在する場合、エラーが発生します。存在しない場合、新しいRunが作成されます。 - `"must"`: 指定された`id`のRunが存在する場合、最後のステップから再開します。存在しない場合、エラーが発生します。 - `"auto"`: このマシンでクラッシュした以前のRunを自動的に再開し、存在しない場合は新しいRunを開始します。 - `True`: 廃止予定です。代わりに`"auto"`を使用してください。 - `False`: 廃止予定です。デフォルトの動作（`resume`を未設定のままにする）を使用して常に新しいRunを開始します。注意: `resume`が設定されている場合、`fork_from`および`resume_from`は使用できません。`resume`が未設定の場合、システムは常に新しいRunを開始します。詳細は[Runの再開に関するガイド](https://docs.wandb.com/guides/runs/resuming)をご覧ください。 |
|  `resume_from` |  前のRunのある時点からRunを再開するための形式`{run_id}?_step={step}`で指定します。これにより、中間ステップでRunにログされた履歴を切り詰め、そのステップからログを再開することができます。ターゲットRunは同じプロジェクト内になければなりません。`id`引数も提供された場合、`resume_from`引数が優先されます。`resume`、`resume_from`、`fork_from`は一緒に使用できません。それらのいずれかのみを使用できます。注意: この機能はベータ版であり、将来的に変更される可能性があります。 |
|  `fork_from` |  前のRunのある時点から新しいRunをフォークするための形式`{id}?_step={step}`で指定します。これは、ターゲットRunの履歴で指定されたステップからログを再開する新しいRunを作成します。ターゲットRunは現在のプロジェクトの一部でなければなりません。`id`引数も提供された場合、それは`fork_from`引数と異なる必要があり、それらが同じ場合はエラーが発生します。`resume`、`resume_from`、`fork_from`は一緒に使用できません。それらのいずれかのみを使用できます。注意: この機能はベータ版であり、将来的に変更される可能性があります。 |
|  `save_code` |  実験の再現性を援助し、UIでRun間のコード比較を可能にするために、メインスクリプトまたはノートブックをW&Bに保存できるようにします。デフォルトでは無効ですが、[設定ページ](https://wandb.ai/settings)でデフォルトを有効にすることができます。 |
|  `tensorboard` |  廃止予定です。代わりに`sync_tensorboard`を使用してください。 |
|  `sync_tensorboard` |  W&BログをTensorBoardやTensorBoardXから自動的に同期し、W&B UIで見るために関連するイベントファイルを保存します。関連するイベントファイルを保存します。（デフォルト: `False`） |
|  `monitor_gym` |  OpenAI Gymを使用する際に環境のビデオを自動的にログします。詳細は[Gymインテグレーションに関するガイド](https://docs.wandb.com/guides/integrations/openai-gym)をご覧ください。 |
|  `settings` |  Runの詳細な設定を持つ辞書または`wandb.Settings`オブジェクトを指定します。 |

| 戻り値 |  |
| :--- | :--- |
|  現在のRunのハンドルである`Run`オブジェクト。このオブジェクトを使用して、データをログしたり、ファイルを保存したり、Runを終了するなどの操作を行うことができます。[Run API](https://docs.wandb.ai/ref/python/sdk/classes/run/)の詳細をご覧ください。 |

| 例外 |  |
| :--- | :--- |
|  `Error` |  Runの初期化中に不明または内部エラーが発生した場合。 |
|  `AuthenticationError` |  ユーザーが有効な資格情報を提供できなかった場合。 |
|  `CommError` |  W&Bサーバーとの通信に問題があった場合。 |
|  `UsageError` |  ユーザーが関数に無効な引数を提供した場合。 |
|  `KeyboardInterrupt` |  ユーザーがRunの初期化プロセスを中断した場合。 |
