# init



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/sdk/wandb_init.py#L920-L1182)


新しいrunを開始し、追跡してW&Bに記録します。

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




MLトレーニング開発フローでは、`wandb.init()`をトレーニングスクリプトと評価スクリプトの開始部分に追加することができます。これにより、各要素はW&B内のrunとして追跡されます。

`wandb.init()`は新しいバックグラウンドプロセスを生成し、データをrunに記録します。また、デフォルトでデータはwandb.aiと同期するため、ライブで可視化を確認できます。

`wandb.log()`でデータを記録する前に、`wandb.init()`を呼び出して、runを開始します。

```python
import wandb

wandb.init()
# ... calculate metrics, generate media
wandb.log({"accuracy": 0.9})
```

`wandb.init()` returns a run object, and you can also access the run object
via `wandb.run`:

```python
import wandb

run = wandb.init()

assert run is wandb.run
```

スクリプトの最後にwandb.finishが自動的に呼び出され、runを終了させてクリーンアップします。ただし、wandb.initを子プロセスから呼び出した場合、子プロセスの終了時にwandb.finishを明示的に呼び出す必要があります。

詳細な例を含むwandb.init()の使用方法の詳細については、[ガイドとよくある質問](../../guides/track/launch)をご覧ください。

| Arguments | |
| :--- | :--- |
| `project` | （str、オプション）新しいrunの送信先のプロジェクト名。プロジェクトを指定しない場合、runは`未分類`プロジェクトに保存されます。 |
| `entity` |（str、オプション）エンティティは、runの送信先のユーザー名またはチーム名になります。このエンティティは、runを送信する前に終了する必要があるため、runを記録する前に、UIでアカウントまたはチームを必ず作成してください。エンティティを指定しないと、runはデフォルトエンティティ（通常はあなたのユーザー名）に送信されます。設定でデフォルトエンティティを変更します（`default location to create new projects（新規プロジェクトを作成するデフォルトロケーション）`の下）。|
| `config` | （dict、argparse、absl.flags、str、オプション）これは、入力をジョブに保存するための、辞書のようなオブジェクト`wandb.config`を設定します。これは、データ前処理ジョブ用のモデルまたは設定用のハイパーパラメーターに似ています。configはUI内のテーブルに表示され、runのグループ化、フィルターおよび並べ替えなどに使用できます。キーの名前に`.`（ピリオド）を含めてはいけません。また、値は10 MB未満にする必要があります。dict、argparseまたはabsl.flagsの場合：キー値のペアがwandb.configオブジェクトに読み込まれます。strの場合：その名前でyamlファイルを探し、そのファイルからconfigをwandb.configオブジェクトに読み込みます。 |
| `save_code` | （bool、オプション）これをオンにして、メインスクリプトまたはノートブックをW&Bに保存します。これは実験の再現可能性を改善し、UI内の複数の実験でコードを区別するのに役立ちます。デフォルトでこれはオフになっていますが、[設定ページ](https://wandb.ai/settings)でデフォルトビヘイビアをオンに切り替えることができます。|
| `group` | （str、オプション）グループを指定し、個々のrunをより大規模な実験に構造化します。たとえば交差検証を行っている、またはさまざまなテストセットに対してモデルのトレーニングと評価を行う複数のジョブがあるとします。グループによって、複数のrunを1つの大きい組織にまとめることができます。UIでこれのオンとオフを切り替えることができます。詳細については、[runのグループ化に関するガイド](https://docs.wandb.com/guides/runs/grouping)を参照してください。 |
| `job_type` | （str、オプション）runのタイプを指定します。これは、グループを使って、複数のrunをまとめて大規模な実験にグループ化する時に役立ちます。たとえば、trainやevalなどのジョブタイプの複数のジョブがグループ内にある場合です。これを設定することで、UIで類似のrunをまとめて簡単にフィルターしたりグループ化したりできます。これにより、合理的な比較を実行できます。|
| `tags` | （list、オプション）文字列のリスト。UI内でこのrunのタグのリストが生成されます。タグは複数のrunをまとめる際や、`ベースライン`や`プロダクション`などの一時的なラベルを適用する際に役立ちます。UIでタグを簡単に追加/削除したり、特定のタグが付いたrunをフィルターしたりすることができます |
| `name` | （str、オプション）このrunの短い表示名。この名前でUIでこのrunを識別します。デフォルトで、ランダムな2ワードの名前が生成されます。これにより、テーブルからグラフまで、runを簡単に相互参照することができます。このrun名を短くすることで、グラフの凡例とテーブルが読みやすくなります。ハイパーパラメーターの保存場所を探している場合、configに保存することをお勧めします。 |
| `notes` | （str、オプション）runの詳細な説明。gitの-mコミットメッセージと同様です。これによって、このrunを実行した時に何をしていたか思い出すことができます。 |
| `dir` | （strまたはpathlib.Path、オプション）メタデータが保存されるディレクトリーへの絶対パス。アーティファクトでdownload()を呼び出すと、これは、ダウンロードされたファイルが保存されるディレクトリーになります。デフォルトで、これは`./wandb`ディレクトリーです。 |
| `resume` | （bool、str、オプション）再開の振る舞いを設定します。オプション："allow"、"must"、"never"、"auto"またはNone。デフォルトはNoneです。事例：- None（デフォルト）：新しいrunが以前のrunと同じIDを持つ場合、このrunはそのデータを上書きします。- "auto"（またはTrue）：このマシン上の以前のrunがクラッシュした場合、自動的に再開します。それ以外の場合、新しいrunを開始します。- "allow"：idが`init (id="UNIQUE_ID`)`または`WANDB_RUN_ID="UNIQUE_ID"`で設定され、以前のrunと同じである場合、wandbはそのidを持つrunを自動的に再開します。それ以外の場合、新しいrunを開始します。- "never"：idが`init (id="UNIQUE_ID`)`または`WANDB_RUN_ID="UNIQUE_ID"`で設定され、以前のrunと同じである場合、wandbはクラッシュします。- "must"：idが`init (id="UNIQUE_ID`)`または`WANDB_RUN_ID="UNIQUE_ID"`で設定され、以前のrunと同じである場合、wandbはそのidを持つrunを自動的に再開します。それ以外の場合、wandbはクラッシュします。詳細は、(、runの再開に関するガイド) を参照してください。 |
| `reinit` | （bool、オプション）同じプロセス内で複数のwandb.init()呼び出しを許可します。（デフォルト：False） |
| `magic` | （bool、dict、またはstr、オプション）boolはスクリプトの自動計測を試すかどうかを制御し、wandbコードをさらに追加する必要なく、runの基本事項をキャプチャします。（デフォルト：False）dict、json文字列、またはyamlファイル名を渡すこともできます。 |
| `config_exclude_keys` | （list、オプション）wandb.configから除外される文字列キー。|
| `config_include_keys` |（list、オプション）wandb.configに含める文字列キー。 |
| `anonymous` | （str、オプション）匿名データロギングを制御します。オプション：- "never"（デフォルト）：匿名runを誤って作成しないように、runを追跡する前にW&Bアカウントをリンクする必要があります。- "allow"：ログインユーザーは自分のアカウントでrunを追跡できますが、スクリプトを実行している、W&Bアカウントを持たない他のユーザーはUI内のグラフを表示することができます。- "must"：サインアップされたユーザーアカウントではなく、匿名アカウントにrunを送信します。 |
| `mode` | （str、オプション）"online"、"offline"または"disabled”になります。デフォルトはonlineです。 |
| `allow_val_change` | (（bool、オプション）キーを一度設定した後、config値の変更を許可するかどうか。デフォルトで、config値が上書きされると例外を投げます。トレーニング中に複数回、さまざまな学習速度などを追跡したい場合、代わりにwandb.log()を使います。（デフォルト：スクリプトではFalse、JupyterではTrue） |
| `force` | （bool、オプション）Trueの場合、およびユーザーがW&Bにログインしていない場合、スクリプトがクラッシュします。Falseの場合、およびユーザーがW&Bにログインしていない場合、これによってスクリプトがオフラインモードで実行します。（デフォルト：False） |
| `sync_tensorboard` | （bool、オプション）tensorboardまたはtensorboardXからwandbログを同期させ、関連するイベントファイルを保存します。（デフォルト：False） |
| `monitor_gym` | （bool、オプション）OpenAI Gymの使用時に、環境の動画を自動的に記録します。（デフォルト：False）この統合に関するガイドを参照してください。 |
| `id` | str、オプション）再開に使用される、このrunの一意のID。これはプロジェクト内で一意である必要があります。runを削除すると、そのIDを再利用することはできません。nameフィールドを使って説明的な名前を付けます。またはconfigを使ってハイパーパラメーターを保存し、複数のrunの間で比較します。IDに以下の特殊文字を含めることはできません [runの再開に関するガイド](https://docs.wandb.com/guides/runs/resuming)を参照してください。 |



#### 例:


### runの記録場所を設定

runが記録される場所を変更することができます。gitで組織、レポジトリおよびブランチを変更するのと同様です:
```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### runに関するメタデータをconfigに追加​

辞書スタイルのオブジェクトをconfigキーワード引数として渡し、ハイパーパラメーターなどのメタデータをrunに追加します。

```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| 以下を返します： | |
| :--- | :--- |
| `Run`オブジェクト。 |

