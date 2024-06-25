
# init

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_init.py#L924-L1186' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

新しいrunを開始して、データを追跡しW&Bにログします。

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

MLトレーニングパイプラインでは、`wandb.init()`をトレーニングスクリプトの最初と評価スクリプトの最初に追加し、それぞれがW&Bのrunとして追跡されるようにすることができます。

`wandb.init()`はバックグラウンドでデータをrunにログする新しいプロセスを生成し、デフォルトではwandb.aiにデータを同期します。これによりライブ可視化が可能になります。

`wandb.init()`を呼び出して、`wandb.log()`でデータをログする前にrunを開始します：

```python
import wandb

wandb.init()
# ... メトリクスを計算し、メディアを生成
wandb.log({"accuracy": 0.9})
```

`wandb.init()`はrunオブジェクトを返しますが、`wandb.run`を通じてrunオブジェクトにアクセスすることもできます：

```python
import wandb

run = wandb.init()

assert run is wandb.run
```

スクリプトの最後に、`wandb.finish`が自動的に呼び出され、runを完了しクリーンアップします。ただし、子プロセスから`wandb.init`を呼び出した場合は、子プロセスの終了時に明示的に`wandb.finish`を呼び出す必要があります。

`wandb.init()`の使用方法の詳細な例については、[ガイドとFAQ](https://docs.wandb.ai/guides/track/launch)をご覧ください。

| 引数 |  |
| :--- | :--- |
|  `project` |  (str, オプション) 新しいrunを送信するプロジェクトの名前。プロジェクトが指定されていない場合、runは "Uncategorized" のプロジェクトに配置されます。 |
|  `entity` |  (str, オプション) エンティティは、runを送信するユーザー名またはチーム名です。このエンティティは、runを送信する前に存在する必要があるため、アカウントまたはチームをUIで作成してからログを開始してください。エンティティを指定しない場合、runはデフォルトのエンティティに送信されます。デフォルトのエンティティは[設定](https://wandb.ai/settings)の「新しいプロジェクトを作成するデフォルトの場所」で変更できます。 |
|  `config` |  (dict, argparse, absl.flags, str, オプション) これは`wandb.config`を設定し、モデルのハイパーパラメーターやデータプロセッシングジョブの設定など、ジョブへの入力を保存するための辞書のようなオブジェクトです。コンフィグはUIのテーブルに表示され、runのグループ化、フィルタリング、ソートに使用できます。キーの名前には `.` を含めず、値は10MB以下にする必要があります。辞書、挙動解析またはabsl.flagsの場合：キーと値のペアを`wandb.config`オブジェクトにロードします。文字列の場合：その名前のyamlファイルを探し、そのファイルからコンフィグを`wandb.config`オブジェクトにロードします。 |
|  `save_code` |  (bool, オプション) メインスクリプトまたはノートブックをW&Bに保存します。これは実験の再現性を向上させ、UIでのexperiment間のコードの差分を確認するために有用です。デフォルトではオフですが、[設定ページ](https://wandb.ai/settings)でデフォルト動作をオンに切り替えられます。 |
|  `group` |  (str, オプション) 個々のrunを大きな実験として整理するためのグループを指定します。たとえば、交差検証を行う場合や異なるテストセットに対してモデルをトレーニングおよび評価する複数のジョブがある場合などです。グループを使用すると、runを一緒に整理し、大きな全体として見る方法が提供され、UIでこれをオンオフできます。詳細は[runのグループ化ガイド](https://docs.wandb.com/guides/runs/grouping)をご覧ください。 |
|  `job_type` |  (str, オプション) runのタイプを指定します。これは、グループを使用してrunを大きな実験にまとめる場合に役立ちます。たとえば、グループ内に複数のジョブがあり、トレーニングや評価などのジョブタイプがある場合です。これを設定すると、UIで同様のrunをフィルターおよびグループ化して比較するのが簡単になります。 |
|  `tags` |  (リスト, オプション) このrunのUI上のタグリストを入力する文字列のリスト。タグはrunを整理したり、「ベースライン」や「プロダクション」のような一時的なラベルを適用するために便利です。UIでタグを簡単に追加および削除したり、特定のタグを持つrunのみにフィルタリングできます。runを再開すると、そのタグは`wandb.init()`に渡されたタグで上書きされます。既存のタグを上書きせずに再開されたrunにタグを追加するには、`wandb.init()`の後で`run.tags += ["new_tag"]`を使用します。 |
|  `name` |  (str, オプション) このrunをUIで識別するための短い表示名。デフォルトでは、runを簡単に参照できるようにランダムな二語の名前が生成されます。このrunの名前を短くしておくことで、チャートの凡例およびテーブルが読みやすくなります。ハイパーパラメーターを保存する場所を探している場合は、configに保存することをお勧めします。 |
|  `notes` |  (str, オプション) gitの`-m`コミットメッセージのように、runの長い説明。このrunを実行したときに何をしていたかを覚えておくのに役立ちます。 |
|  `dir` |  (strまたはpathlib.Path, オプション) メタデータが保存されるディレクトリーへの絶対パス。artifactで`download()`を呼び出すと、ダウンロードされたファイルがこのディレクトリーに保存されます。デフォルトでは、`./wandb`ディレクトリーです。 |
|  `resume` |  (bool, str, オプション) 再開動作を設定します。選択肢: `"allow"`, `"must"`, `"never"`, `"auto"`または`None`。デフォルトでは`None`。ケース: - `None` (デフォルト): 新しいrunが以前のrunと同じIDを持つ場合、このrunはそのデータを上書きします。 - `"auto"` (または`True`): このデバイスで以前のrunがクラッシュしていた場合、自動的に再開します。それ以外の場合、新しいrunを開始します。 - `"allow"`: `init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`でIDが設定され、以前のrunと同一である場合、wandbは自動的にそのIDでrunを再開します。それ以外の場合、新しいrunを開始します。 - `"never"`: `init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`でIDが設定され、以前のrunと同一である場合、wandbはクラッシュします。 - `"must"`: `init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`でIDが設定され、以前のrunと同一である場合、wandbは自動的にそのIDでrunを再開します。それ以外の場合、wandbはクラッシュします。詳細は[runの再開ガイド](https://docs.wandb.com/guides/runs/resuming)をご覧ください。 |
|  `reinit` |  (bool, オプション) 同じプロセスで複数の`wandb.init()`呼び出しを許可します。(デフォルト: `False`) |
|  `magic` |  (bool, dict, またはstr, オプション) スクリプトを自動でインストルメントし、追加のwandbコードを追加せずにrunの基本的な詳細をキャプチャするかどうかを制御します。(デフォルト: `False`) dict、json文字列、またはyamlファイル名を渡すこともできます。 |
|  `config_exclude_keys` |  (リスト, オプション) `wandb.config`から除外する文字列キー。 |
|  `config_include_keys` |  (リスト, オプション) `wandb.config`に含める文字列キー。 |
|  `anonymous` |  (str, オプション) 匿名データログを制御します。オプション: - `"never"` (デフォルト): runを追跡する前にW&Bアカウントにリンクする必要があり、誤って匿名runを作成しないようにします。 - `"allow"`: ログインしているユーザーがrunをアカウントで追跡できますが、W&Bアカウントなしでスクリプトを実行している場合は、UIでチャートを表示できます。 - `"must"`: runをサインアップ済みユーザーアカウントではなく、匿名アカウントに送信します。 |
|  `mode` |  (str, オプション) `"online"`, `"offline"`または`"disabled"`に設定できます。デフォルトはオンラインです。 |
|  `allow_val_change` |  (bool, オプション) キーを一度設定した後にconfigの値を変更するかどうかを制御します。デフォルトでは、config値を上書きすると例外が発生します。学習率のような変動する値をトレーニングの複数のタイミングで追跡したい場合は、代わりに`wandb.log()`を使用してください。(スクリプトではデフォルト: `False`, Jupyterでは`True`) |
|  `force` |  (bool, オプション) `True`の場合、ユーザーがW&Bにログインしていないとスクリプトがクラッシュします。`False`の場合、ユーザーがW&Bにログインしていない場合でもスクリプトはオフラインモードで実行されます。(デフォルト: `False`) |
|  `sync_tensorboard` |  (bool, オプション) tensorboardまたはtensorboardXからwandbログを同期し、関連するイベントファイルを保存します。(デフォルト: `False`) |
|  `monitor_gym` |  (bool, オプション) OpenAI Gymを使用する際に環境のビデオを自動的にログします。(デフォルト: `False`) [このインテグレーションのガイド](https://docs.wandb.com/guides/integrations/openai-gym)をご覧ください。 |
|  `id` |  (str, オプション) このrunの一意なID。再開に使用されます。プロジェクト内で一意である必要があります。runを削除すると、IDを再利用できません。短い説明的な名前には`name`フィールドを使用し、run間で比較するためにハイパーパラメーターを保存するには`config`を使用します。IDには次の特殊文字を含めることはできません: `/\#?%:`。詳細は[runの再開ガイド](https://docs.wandb.com/guides/runs/resuming)をご覧ください。 |
|  `fork_from` |  (str, オプション) 前のrunの特定のステップから新しいrunをフォークするための{run_id}?_step={step}形式の文字列。指定された時点でフォークされた新しいrunを作成し、指定されたmomentでのrunのログ履歴をピックアップします。対象のrunは現在のプロジェクト内に存在する必要があります。例: `fork_from="my-run-id?_step=1234"`。[runのフォーキングガイド](https://docs.wandb.com/guides/runs/forking)をご覧ください。 |
| `resume_from` | (str, optional) 前のrunの特定のステップから新しいrunを"巻き戻す"ための{run_id}?_step={step}形式の文字列。指定されたステップまでのログ履歴を保持しながら、新しいrunをそのステップに巻き戻します。対象のrunは現在のプロジェクト内に存在する必要があります。例: `resume_from="my-run-id?_step=1234"`。[runの巻き戻しガイド](https://docs.wandb.com/guides/runs/rewind)をご覧ください。|

### 例:

### Runがログされる場所を設定する

Runがログされる場所を変更できます。ちょうどgitで組織、リポジトリ、ブランチを変更するように:

```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### Runのコンフィグにメタデータを追加する

辞書スタイルのオブジェクトを`config`キーワード引数として渡して、runのメタデータ（ハイパーパラメーターなど）を追加します。

```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| 例外 |  |
| :--- | :--- |
|  `Error` |  run初期化中に未知のエラーまたは内部エラーが発生した場合。 |
|  `AuthenticationError` |  ユーザーが有効な資格情報を提供できなかった場合。 |
|  `CommError` |  WandBサーバーとの通信に問題があった場合。 |
|  `UsageError` |  ユーザーが無効な引数を指定した場合。 |
|  `KeyboardInterrupt` |  ユーザーがrunを中断した場合。 |

| 戻り値 |  |
| :--- | :--- |
|  `Run`オブジェクト。 |