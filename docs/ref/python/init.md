# init

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_init.py#L924-L1186' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

新しいRunを開始して、W&Bにトラックしログを記録します。

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

MLのトレーニングパイプラインでは、`wandb.init()`をトレーニングスクリプトの冒頭や評価スクリプトの冒頭に追加することができ、それぞれがW&B内で1つのRunとしてトラックされます。

`wandb.init()`はバックグラウンドで新しいプロセスを生成し、データをRunにログ記録します。また、デフォルトでデータをwandb.aiへ同期し、リアルタイムの可視化を提供します。

データを`wandb.log()`でログする前に、Runを開始するために`wandb.init()`を呼び出します：

```python
import wandb

wandb.init()
# ... メトリクスを計算し、メディアを生成
wandb.log({"accuracy": 0.9})
```

`wandb.init()`はRunオブジェクトを返し、`wandb.run`を通じてRunオブジェクトにアクセスすることもできます：

```python
import wandb

run = wandb.init()

assert run is wandb.run
```

スクリプトの終了時に、`wandb.finish`を自動的に呼び出してRunを完了し、クリーンアップします。しかし、子プロセスから`wandb.init`を呼び出した場合は、子プロセスの終了時に明示的に`wandb.finish`を呼び出す必要があります。

`wandb.init()`の詳細な使い方や具体例については、[ガイドとFAQ](https://docs.wandb.ai/guides/track/launch) をチェックしてください。

| 引数 |  |
| :--- | :--- |
|  `project` |  (str, optional) 新しいRunを送信するプロジェクトの名前。プロジェクトが指定されていない場合、Runは「未分類」のプロジェクトに配置されます。 |
|  `entity` |  (str, optional) Runを送信するユーザー名またはチーム名。このエンティティは事前に存在している必要があるため、アカウントやチームをUIで作成してからRunを記録するようにしてください。エンティティを指定しない場合、Runはデフォルトのエンティティに送信されます。デフォルトエンティティは[設定](https://wandb.ai/settings)で「新しいプロジェクトを作成するデフォルト位置」の下で変更できます。 |
|  `config` |  (dict, argparse, absl.flags, str, optional) `wandb.config`を設定し、モデルのハイパーパラメーターやデータプロセッシングジョブの設定など、ジョブへの入力を保存する辞書風のオブジェクト。設定はUIのテーブルに表示され、Runのグループ、フィルター、ソートに使用できます。キーは名前に`.` を含めないようにし、値は10MB未満にすること。辞書、argparse、absl.flagsの場合、キーと値のペアを`wandb.config`オブジェクトにロードします。文字列の場合、その名前のyamlファイルを探し、それを`wandb.config`オブジェクトにロードします。 |
|  `save_code` |  (bool, optional) このオプションをオンにすると、メインのスクリプトまたはノートブックをW&Bに保存します。これは実験の再現性を向上させるために有用で、UIで実験間のコードを比較できます。デフォルトではオフですが、[設定ページ](https://wandb.ai/settings)でデフォルトの振る舞いをオンに変更できます。 |
|  `group` |  (str, optional) 個々のRunを大規模な実験にまとめるためのグループを指定します。例えば、交差検証を行う場合や、異なるテストセットに対してモデルをトレーニングおよび評価する複数のジョブがある場合など。このグループを使用することで、Runを全体の一部として整理できます。詳細については、[Runのグループ化ガイド](https://docs.wandb.com/guides/runs/grouping) を参照してください。 |
|  `job_type` |  (str, optional) Runの種類を指定し、グループでRunをまとめるときに便利です。例えば、グループ内の複数のジョブがあり、それぞれのジョブタイプが「トレイン」や「評価」であるとします。これを設定することで、UIで同様のRunをフィルターおよびグループ化して比較しやすくなります。 |
|  `tags` |  (list, optional) このRunにタグのリストを追加します。タグはRunを整理するのに役立ち、「ベースライン」や「プロダクション」のような一時的なラベルを適用できます。UIで簡単にタグを追加および削除したり、特定のタグを持つRunのみにフィルターすることができます。Runを再開する場合、そのRunのタグは`wandb.init()`に渡したタグで上書きされます。既存のタグを上書きせずにRunにタグを追加したい場合は、`wandb.init()`後に`run.tags += ["new_tag"]`を使用してください。 |
|  `name` |  (str, optional) このRunの短い表示名。これによって、UIでこのRunを識別します。デフォルトではランダムな2単語の名前が生成され、テーブルからチャートへのクロスリファレンスが容易になります。これらのRun名を短く保つことで、チャートの凡例やテーブルが読みやすくなります。ハイパーパラメーターを保存する場所を探している場合は、設定に保存することをお勧めします。 |
|  `notes` |  (str, optional) gitの`-m`コミットメッセージのように、Runの長い説明。Runを実行した際の状況を思い出すのに役立ちます。 |
|  `dir` |  (str or pathlib.Path, optional) メタデータが保存されるディレクトリーへの絶対パス。アーティファクトに対して`download()`を呼び出すと、このディレクトリーにダウンロードされたファイルが保存されます。デフォルトでは、`./wandb`ディレクトリーです。 |
|  `resume` |  (bool, str, optional) 再開の振る舞いを設定します。オプション: `"allow"`, `"must"`, `"never"`, `"auto"`または`None`。デフォルトは`None`です。ケース: - `None` (デフォルト): 新しいRunが以前のRunと同じIDを持っている場合、このRunはそのデータを上書きします。 - `"auto"` (または `True`): このマシンで以前のRunがクラッシュした場合、自動的に再開します。そうでない場合は、新しいRunを開始します。 - `"allow"`: `init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`を使用してIDが設定され、以前のRunと一致する場合、そのIDのRunを自動的に再開します。そうでない場合は、新しいRunを開始します。 - `"never"`: `init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`を使用してIDが設定され、以前のRunと一致する場合、wandbはクラッシュします。 - `"must"`: `init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`を使用してIDが設定され、以前のRunと一致する場合、そのIDのRunを自動的に再開します。そうでない場合は、wandbはクラッシュします。詳細については、[Runの再開に関するガイド](https://docs.wandb.com/guides/runs/resuming) を参照してください。 |
|  `reinit` |  (bool, optional) 同一プロセスで複数の`wandb.init()`呼び出しを許可します。(デフォルト: `False`) |
|  `magic` |  (bool, dict, or str, optional) スクリプトの基本的な詳細を追加のwandbコードなしでキャプチャするための自動インスツルメント設定。(デフォルト: `False`) dictやjson文字列、yamlファイル名を渡すこともできます。 |
|  `config_exclude_keys` |  (list, optional) `wandb.config`から除外する文字列キー。 |
|  `config_include_keys` |  (list, optional) `wandb.config`に含める文字列キー。 |
|  `anonymous` |  (str, optional) 匿名データログを制御します。オプション: - `"never"` (デフォルト): Runをトラックする前にW&Bアカウントをリンクする必要があります。偶然に匿名Runを作成しないようにするためです。 - `"allow"`: ログインしたユーザーは自身のアカウントでRunをトラックできますが、W&Bアカウントを持たない人がスクリプトを実行してもUIでチャートを確認できます。 - `"must"`: Runをサインアップしたユーザーアカウントではなく、匿名アカウントに送信します。 |
|  `mode` |  (str, optional) `"online"`, `"offline"`または`"disabled"`が選べます。デフォルトはオンライン。 |
|  `allow_val_change` |  (bool, optional) キーを一度設定した後にconfigの値を変更できるかどうか。デフォルトでは、config値が上書きされると例外を投げます。例えば、トレーニング中に学習率が変動するようなものをトラックしたい場合は、代わりに`wandb.log()`を使用してください。(デフォルト: スクリプトでは`False`, Jupyterでは`True`) |
|  `force` |  (bool, optional) `True`の場合、ユーザーがW&Bにログインしていないとスクリプトがクラッシュします。`False`の場合、ユーザーがW&Bにログインしていないとオフラインモードでスクリプトを実行します。(デフォルト: `False`) |
|  `sync_tensorboard` |  (bool, optional) tensorboardやtensorboardXからwandbログを同期し、関連するイベントファイルを保存します。(デフォルト: `False`) |
|  `monitor_gym` |  (bool, optional) OpenAI Gymを使用しているときに環境のビデオを自動的にログします。(デフォルト: `False`) このインテグレーションに関するガイドは[こちら](https://docs.wandb.com/guides/integrations/openai-gym) を参照してください。 |
|  `id` |  (str, optional) このRunのための一意のID。再開に使用します。プロジェクト内で一意である必要があり、Runを削除してもIDを再利用することはできません。短い説明名には`name`フィールドを使用し、Run全体にわたる比較に使用するハイパーパラメーターの保存には`config`を使用してください。このIDには次の特殊文字を含めることはできません： `/\#?%:`。詳細については、[Runの再開に関するガイド](https://docs.wandb.com/guides/runs/resuming) を参照してください。 |
|  `fork_from` |  (str, optional) {run_id}?_step={step}の形式で以前のRunのある瞬間を表す文字列。指定されたRunの指定された時点から新しいRunをフォークして作成します。ターゲットのRunは現在のプロジェクト内にある必要があります。例：`fork_from="my-run-id?_step=1234"`。|

#### 例:

### Runのログ場所を設定

Runのログ場所を変更できます。これはgitでの組織、リポジトリ、ブランチの変更と同様です：

```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### Runに関するメタデータをconfigに追加

辞書風のオブジェクトを`config`キーワード引数として渡し、ハイパーパラメーターなどのメタデータをRunに追加します。

```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| 発生しうるエラー |  |
| :--- | :--- |
|  `Error` |  Runの初期化の際に不明または内部エラーが発生した場合。 |
|  `AuthenticationError` |  ユーザーが有効な認証情報を提供できなかった場合。 |
|  `CommError` |  WandBサーバーとの通信に問題があった場合。 |
|  `UsageError` |  ユーザーが無効な引数を提供した場合。 |
|  `KeyboardInterrupt` |  ユーザーがRunを中断した場合。 |

