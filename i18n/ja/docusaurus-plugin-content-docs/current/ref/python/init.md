# init

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_init.py#L913-L1184)

W&Bにトラッキングとログを行う新しいrunを開始します。

```python
init(
 job_type: Optional[str] = None,
 dir: Union[str, pathlib.Path, None] = None,
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
 settings: Union[Settings, Dict[str, Any], None] = None
) -> Union[Run, RunDisabled, None]
```
MLトレーニング開発フローで、トレーニングスクリプトの最初や評価スクリプトの最初に`wandb.init()`を追加することで、それぞれの部分がW&Bでrunとしてトラッキングされます。

`wandb.init()`は、新しいバックグラウンドプロセスを生成してデータをrunにログするだけでなく、デフォルトでデータをwandb.aiに同期させ、リアルタイムのデータ可視化を表示できます。
以下のMarkdownテキストを翻訳してください。日本語に翻訳して、それ以外のことは何も言わずに翻訳したテキストだけを返してください。テキスト:

`wandb.init()`を呼び出して`wandb.log()`でデータをログする前にrunを開始します：

```python
import wandb

wandb.init()
こちらが翻訳すべきMarkdownテキストの一部です。日本語に翻訳してください。それ以外のことは何も言わないで、翻訳したテキストのみを返してください。テキスト：

# ... メトリクスを計算し、メディアを生成する
wandb.log({"accuracy": 0.9})
```

`wandb.init()`はrunオブジェクトを返し、`wandb.run`を使ってもrunオブジェクトにアクセスできます。

```python
import wandb
以下は、Markdownテキストの一部を翻訳してください。日本語に翻訳して、それ以外のことは言わずに翻訳されたテキストのみを返してください。テキスト：

```
run = wandb.init()

assert run is wandb.run
```

スクリプトの最後で、自動的に`wandb.finish`を呼び出して、実行を最終化し、クリーンアップします。ただし、子プロセスから`wandb.init`を呼び出す場合は、子プロセスの終わりで明示的に`wandb.finish`を呼び出す必要があります。
`wandb.init()`の使用方法や詳細な例については、
[ガイドとFAQ](https://docs.wandb.ai/guides/track/launch)をご覧ください。


| 引数 | |
| :--- | :--- |
| `project` | (str, オプション) 新しいrunを送信するプロジェクトの名前。プロジェクトが指定されていない場合、runは"Uncategorized"プロジェクトに入れられます。 |
| `entity` | (str, オプション) 送信先のユーザー名やチーム名。実行を送信する前に、このエンティティを作成する必要がありますので、ログの送信を開始する前にUIでアカウントやチームを作成してください。エンティティが指定されていない場合、runは通常のユーザー名であるデフォルトのエンティティに送信されます。デフォルトのエンティティは[設定](https://wandb.ai/settings)の「新しいプロジェクトを作成する場所」で変更できます。 |
| `config` | (dict, argparse, absl.flags, str, オプション) これにより、`wandb.config`が設定されます。これは、モデルのハイパーパラメーターやデータ前処理ジョブの設定など、ジョブに入力するための辞書のようなオブジェクトです。設定はUIのテーブルに表示され、実行をグループ化、フィルター、並べ替えることができます。キーには`.`を含めず、値は10 MB以下にしてください。もしdict, argparse, absl.flagsの場合：`wandb.config`オブジェクトにキーバリューペアを読み込む。strの場合：その名前のyamlファイルを検索し、そのファイルから`wandb.config`オブジェクトに設定を読み込む。|
| `save_code` | (bool, オプション) これをオンにすると、メインのスクリプトやノートブックがW&Bに保存されます。これは実験の再現性を向上させ、UIで実験間のコードを比較するために有益です。デフォルトではオフですが、[設定ページ](https://wandb.ai/settings)でデフォルトの動作をオンに切り替えることができます。|
| `group` | (str、オプション) 個々のrunをより大きな実験にまとめるためのグループを指定します。例えば、クロスバリデーションを行っている場合や、異なるテストセットに対してモデルをトレーニングおよび評価する複数のジョブがある場合などです。グループは、runをより大きな単位にまとめる方法を提供し、UIでこれをオン・オフできます。詳細については、[runのグループ化に関するガイド](https://docs.wandb.com/guides/runs/grouping) を参照してください。 |
| `job_type` | (str、オプション) runのタイプを指定します。これは、groupを使用してrunをより大きな実験にまとめる場合に便利です。例えば、グループ内に複数のジョブがあり、train や eval などのジョブタイプがあるかもしれません。これを設定することで、UIで同様のrunをまとめてフィルタリングしやすくなり、同じもの同士を比較できます。 |
| `tags` | (list、オプション) 文字列のリストで、これによってUIでこのrunのタグ一覧が表示されます。タグは、runをまとめたり、「ベースライン」や「プロダクション」のような一時的なラベルを適用するのに便利です。UIでタグを追加・削除したり、特定のタグを持つrunに絞り込むのが簡単です。 |
| `name` | (str、オプション) このrunの短い表示名で、これによってUIでこのrunを識別します。デフォルトでは、ランダムな2単語の名前を生成し、テーブルからチャートへの参照を簡単に行えるようにします。run名を短く保つことで、チャートの凡例やテーブルが読みやすくなります。ハイパーパラメータを保存する場所を探している場合は、configに保存することをお勧めします。
| `notes` | (str, 任意) runのより長い説明文です。gitのcommitメッセージのようなものです。このrunを実行したときに何をしていたかを思い出すのに役立ちます。 |
| `dir` | (str または pathlib.Path, 任意) メタデータが保存されるディレクトリへの絶対パス。アーティファクトの`download()`を呼び出すと、ダウンロードされたファイルが保存されるディレクトリです。デフォルトでは`./wandb`ディレクトリです。 |
| `resume` | (bool, str, 任意) 再開の振る舞いを設定します。オプション: `"allow"`, `"must"`, `"never"`, `"auto"` または `None`。デフォルトは `None`。ケース: - `None` (デフォルト): 新しいrunが以前のrunと同じIDを持っている場合、このrunはそのデータを上書きします。 - `"auto"` (または `True`): このマシン上の前回のrunがクラッシュした場合、自動的に再開します。それ以外の場合は、新しいrunを開始します。 - `"allow"`: idが`init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`で設定され、それが以前のrunと同じ場合、wandbはそのIDを持つrunを自動的に再開します。それ以外の場合、wandbは新しいrunを開始します。 - `"never"`: idが`init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`で設定され、それが以前のrunと同じ場合、wandbはクラッシュします。 - `"must"`: idが`init(id="UNIQUE_ID")`または`WANDB_RUN_ID="UNIQUE_ID"`で設定され、それが以前のrunと同じ場合、wandbはそのIDを持つrunを自動的に再開します。それ以外の場合、wandbはクラッシュします。 詳細は [runの再開に関するガイド](https://docs.wandb.com/guides/runs/resuming) を参照してください。|
| `reinit` | (bool, 任意) 同じプロセスで複数の`wandb.init()`呼び出しを許可します。(デフォルト: `False`) |
| `magic` | (bool, dict, または str, 任意) このbool値は、wandbコードを追加せずにrunの基本的な詳細をキャプチャしようとするかどうかを制御します。(デフォルト: `False`) dict、json文字列、またはyamlファイル名も渡すことができます。 |
| `config_exclude_keys` | (list, 任意) `wandb.config`から除外する文字列キー。 |
| `config_include_keys` | (list, 任意) `wandb.config`に含める文字列キー。 |
| `anonymous` | (str, 任意) 匿名データロギングを制御します。オプション: - `"never"` (デフォルト): runを追跡する前にW&Bアカウントをリンクしておく必要があります。誤って匿名runを作成しないようにします。 - `"allow"`: ログイン済みのユーザーが自分のアカウントでrunを追跡し、W&Bアカウントを持っていない人がUIでグラフを表示できるようにします。 - `"must"`: runを登録済みのユーザーアカウントではなく匿名アカウントに送ります。 |
| `mode` | (str, 任意) `"online"`、`"offline"`、または`"disabled"`にできます。デフォルトはonlineです。 |
| `allow_val_change` | (bool, 任意) 設定値を一度設定したキーを変更できるかどうかを許可します。デフォルトでは、 config値が上書きされた場合に例外をスローします。 トレーニング中に複数回変化する学習率のようなものを追跡したい場合は、代わりに`wandb.log()`を使用してください。(デフォルト: スクリプト内で`False`、Jupyter内で`True`) |
| `force` | (bool, 任意) `True`の場合、W&Bにログインしていないユーザーに対してスクリプトを強制終了します。`False`の場合、W&Bにログインしていないユーザーでもオフラインモードでスクリプトが実行されます。(デフォルト: `False`) |
| `sync_tensorboard` | (bool, 任意) tensorboardまたはtensorboardXからwandbログを同期し、関連するイベントファイルを保存します。(デフォルト: `False`) |
| `monitor_gym` | (bool, 任意) OpenAI Gymを使用している場合、環境のビデオを自動的にログに記録します。(デフォルト: `False`) この統合に関する[当社のガイド](https://docs.wandb.com/guides/integrations/openai-gym)を参照してください。|
| `id` | (str, 任意) このrunの一意のIDで、再開に使用されます。プロジェクト内で一意でなければならず、runを削除した場合、IDを再利用することはできません。短い説明的な名前には`name`フィールドを、ハイパーパラメーターを比較するために`config`を使用してください。IDには次の特殊文字を含めることができません：`/\#?%:`。runの再開に関する[当社のガイド](https://docs.wandb.com/guides/runs/resuming)を参照してください。|
#### 例：

### 実行ログの保存場所を設定する

実行ログの保存場所を変更することができます。これは、gitで組織、リポジトリ、およびブランチを変更するのと同様のことです。
```python
import wandb
以下は翻訳するMarkdownテキストの一部です。これを日本語に翻訳してください。他に何も言わずに、翻訳されたテキストのみ返してください。テキスト：

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```
### Runの設定にメタデータを追加

`config`キーワード引数として辞書型オブジェクトを渡すことで、ハイパーパラメーターなどのメタデータをrunに追加できます。

```python
import wandb
config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| 例外 | |
| :--- | :--- |
| `Error` | runの初期化中に何らかの不明なエラーや内部エラーが発生した場合。 |
| `AuthenticationError` | ユーザーが有効な認証情報を提供できなかった場合。 |
| `CommError` | WandBサーバーとの通信中に問題が発生した場合。 |
| `UsageError` | ユーザーが無効な引数を提供した場合。 |
| `KeyboardInterrupt` | ユーザーがrunを中断した場合。 |
ここに翻訳するMarkdownテキストがあります。日本語に翻訳してください。それ以外のことは何も言わずに、翻訳されたテキストのみを返してください。テキスト：
| 返り値 | |
| :--- | :--- |
| `Run`オブジェクト。 |