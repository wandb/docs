---
title: init()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-init
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py >}}




### <kbd>関数</kbd> `init`

```python
init(
    entity: 'str | None' = None,
    project: 'str | None' = None,
    dir: 'StrPath | None' = None,
    id: 'str | None' = None,
    name: 'str | None' = None,
    notes: 'str | None' = None,
    tags: 'Sequence[str] | None' = None,
    config: 'dict[str, Any] | str | None' = None,
    config_exclude_keys: 'list[str] | None' = None,
    config_include_keys: 'list[str] | None' = None,
    allow_val_change: 'bool | None' = None,
    group: 'str | None' = None,
    job_type: 'str | None' = None,
    mode: "Literal['online', 'offline', 'disabled', 'shared'] | None" = None,
    force: 'bool | None' = None,
    anonymous: "Literal['never', 'allow', 'must'] | None" = None,
    reinit: "bool | Literal[None, 'default', 'return_previous', 'finish_previous', 'create_new']" = None,
    resume: "bool | Literal['allow', 'never', 'must', 'auto'] | None" = None,
    resume_from: 'str | None' = None,
    fork_from: 'str | None' = None,
    save_code: 'bool | None' = None,
    tensorboard: 'bool | None' = None,
    sync_tensorboard: 'bool | None' = None,
    monitor_gym: 'bool | None' = None,
    settings: 'Settings | dict[str, Any] | None' = None
) → Run
```

W&B にトラッキングとログを送る新しい run を開始します。

ML のトレーニング パイプラインでは、トレーニング スクリプトと評価スクリプトの冒頭に `wandb.init()` を追加できます。それぞれが W&B 上で run として追跡されます。

`wandb.init()` は、run へのデータのログ記録のために新しいバックグラウンド プロセスを起動し、既定で https://wandb.ai にデータを同期するので、結果をリアルタイムに確認できます。データのログが終わったら `wandb.Run.finish()` を呼び出して run を終了してください。`run.finish()` を呼び出さない場合、スクリプトの終了時に run は終了します。

Run ID には次の特殊文字を含めないでください `/ \ # ? % :`



**引数:**
 
 - `entity`:  run を記録するユーザー名または Team 名。Entity は事前に存在している必要があるため、run の記録を始める前に UI でアカウントまたは Team を作成してください。指定しない場合は、既定の Entity に記録されます。既定の Entity を変更するには、設定の「Default team」内にある「Default location to create new projects」を更新してください。 
 - `project`:  この run を記録する Project の名前。指定しない場合は、git のルートや現在のプログラム ファイルの場所など、システムに基づくヒューリスティクスで Project 名を推測します。推測できない場合は `"uncategorized"` が既定になります。 
 - `dir`:  実験のログやメタデータ ファイルを保存するディレクトリーへの絶対パス。指定しない場合は `./wandb` ディレクトリーが既定です。なお、これは `download()` 呼び出し時に Artifacts が保存される場所には影響しません。 
 - `id`:  この run の再開に使われる一意の識別子。同じ Project 内で一意である必要があり、run を削除した後に再利用することはできません。短い説明的な名前が必要な場合は `name` を、run 間で比較するためにハイパーパラメーターを保存したい場合は `config` を使ってください。 
 - `name`:  UI に表示されるこの run の短い表示名。既定では 2 語のランダムな名前を生成し、テーブルやチャート間で run を簡単に突き合わせられるようにします。名前を短くしておくと、チャートの凡例やテーブルでの可読性が向上します。ハイパーパラメーターを保存するには `config` フィールドの使用を推奨します。 
 - `notes`:  Git のコミットメッセージのように、この run の詳細な説明。将来この run の目的やセットアップを思い出す助けになる文脈や詳細をここに残してください。 
 - `tags`:  UI でこの run にラベル付けするためのタグのリスト。タグは run の整理や "baseline"、"production" のような一時的な識別子の付与に便利です。UI でタグの追加・削除やタグによるフィルタリングが簡単にできます。run を再開する場合、ここで指定したタグは既存のタグを置き換えます。現在のタグを上書きせずに再開した run にタグを追加するには、`run = wandb.init()` の後に `run.tags += ("new_tag",)` を実行してください。 
 - `config`:  `wandb.config` を設定します。これは、モデルのハイパーパラメーターやデータ 前処理の設定など、run への入力パラメータを保存する辞書風の オブジェクトです。Config は UI の概要ページに表示され、これらのパラメータで run をグループ化・フィルター・ソートできます。キーにはピリオド (`.`) を含めないでください。値は 10 MB 未満である必要があります。辞書、`argparse.Namespace`、または `absl.flags.FLAGS` が渡された場合は、そのキーと値の組がそのまま `wandb.config` に読み込まれます。文字列が渡された場合は YAML ファイルへのパスとみなし、そのファイルからの 設定 値を `wandb.config` に読み込みます。 
 - `config_exclude_keys`:  `wandb.config` から除外する特定のキーのリスト。 
 - `config_include_keys`:  `wandb.config` に含める特定のキーのリスト。 
 - `allow_val_change`:  Config の 値 を初期設定後に変更できるかを制御します。既定では、config の 値 を上書きすると例外が送出されます。学習率のようにトレーニング中に変化する変数を追跡する場合は、代わりに `wandb.log()` の使用を検討してください。スクリプトでは既定で `False`、ノートブック 環境では `True` です。 
 - `group`:  個々の run を大きな実験の一部として整理するためのグループ名。例えばクロスバリデーションや、異なる テストセット で モデル の学習と評価を行う複数ジョブの実行などに有用です。グループ化により、関連する run を UI でまとめて管理でき、統一された実験として結果を切り替えて確認できます。 
 - `job_type`:  run の種類を指定します。大きな実験の一部としてグループ内の run を整理する際に役立ちます。例えば、同じグループの run に "train" や "eval" のようなジョブタイプを付けることができます。ジョブタイプを定義すると、UI で類似の run を簡単にフィルタリング・グループ化でき、直接比較がしやすくなります。 
 - `mode`:  run データの扱いを次のオプションで指定します: 
    - `"online"` (既定): ネットワーク接続があるときに W&B とライブ同期し、可視化をリアルタイム更新します。 
    - `"offline"`: エアギャップやオフラインの 環境 向け。データはローカルに保存され、後で同期できます。将来の同期のために run フォルダーを保持してください。 
    - `"disabled"`: すべての W&B の機能を無効化し、run のメソッドは no-op（何もしない）になります。通常はテストで W&B の処理を回避するために使います。 
    - `"shared"`:（実験的機能）複数のプロセスが、場合によっては異なるマシンから、同じ run に同時にログを送れるようにします。この方式ではプライマリ ノードと 1 台以上のワーカー ノードを用いて、同じ run にデータをログします。プライマリ ノードで run を初期化し、各ワーカー ノードではプライマリ ノードで使った Run ID を使って run を初期化します。 
 - `force`:  スクリプトの実行に W&B へのログインが必須かどうかを決定します。`True` の場合、ユーザー は W&B にログインしている必要があり、そうでない場合スクリプトは先に進みません。`False`（既定）の場合、ログインなしでも実行でき、ユーザー がログインしていないときはオフライン モードに切り替わります。 
 - `anonymous`:  匿名データ ログの制御レベルを指定します。選択肢は次のとおりです: 
    - `"never"`（既定）: run を追跡する前に W&B アカウントとのリンクを求めます。各 run がアカウントに紐づくようにして、意図しない匿名 run の作成を防ぎます。 
    - `"allow"`: ログイン済みのユーザー は自分のアカウントで run を追跡できますが、W&B アカウントのない人がスクリプトを実行した場合でも、UI でチャートやデータを閲覧できるようにします。 
    - `"must"`: ユーザー がログインしている場合でも、run を匿名アカウントに記録することを強制します。 
 - `reinit`:  "reinit" 設定の短縮形。run がアクティブなときに `wandb.init()` がどう振る舞うかを決定します。 
 - `resume`:  指定した `id` を持つ run の再開時の振る舞いを制御します。選択肢は次のとおりです: 
    - `"allow"`: 指定した `id` の run が存在すれば最後の step から再開し、存在しなければ新しい run を作成します。 
    - `"never"`: 指定した `id` の run が存在する場合はエラーを送出します。存在しなければ新しい run を作成します。 
    - `"must"`: 指定した `id` の run が存在すれば最後の step から再開します。存在しなければエラーを送出します。 
    - `"auto"`: このマシン上で前回の run がクラッシュしていれば自動的に再開し、そうでなければ新しい run を開始します。 
    - `True`: 非推奨。代わりに `"auto"` を使ってください。 
    - `False`: 非推奨。既定の挙動（`resume` を未設定のままにする）を使って、常に新しい run を開始してください。`resume` が設定されている場合、`fork_from` と `resume_from` は使用できません。`resume` が未設定のときは、常に新しい run を開始します。 
 - `resume_from`:  以前の run のある時点から再開する場所を `{run_id}?_step={step}` の形式で指定します。run に記録された履歴を中間の step で切り詰め、その step からのログ再開を可能にします。対象の run は同じ Project に属している必要があります。`id` 引数も指定された場合は、`resume_from` が優先されます。`resume`、`resume_from`、`fork_from` は同時に使えず、いずれか 1 つのみ使用できます。この機能はベータ版であり、将来変更される可能性があります。 
 - `fork_from`:  以前の run の履歴の特定の時点から新しい run をフォークする場所を `{id}?_step={step}` の形式で指定します。対象 run の履歴の指定した step からログを再開する新しい run を作成します。対象の run は現在の Project の一部である必要があります。`id` 引数も指定された場合は、`fork_from` 引数と同じであってはならず、同じであればエラーになります。`resume`、`resume_from`、`fork_from` は同時に使えず、いずれか 1 つのみ使用できます。この機能はベータ版であり、将来変更される可能性があります。 
 - `save_code`:  メインの スクリプト または ノートブック を W&B に保存できるようにし、実験の再現性を高め、run 間で コード を比較できるようにします。既定では無効ですが、設定ページで既定を有効に変更できます。 
 - `tensorboard`:  非推奨。代わりに `sync_tensorboard` を使用してください。 
 - `sync_tensorboard`:  TensorBoard または TensorBoardX からの W&B ログを自動的に同期し、W&B の UI で閲覧するための関連するイベント ファイルを保存します。 
 - `saving relevant event files for viewing in the W&B UI. (Default`:  `False`) 
 - `monitor_gym`:  OpenAI Gym を使用する際に、環境の動画を自動で記録します。 
 - `settings`:  run の高度な設定を含む辞書または `wandb.Settings` オブジェクトを指定します。 



**戻り値:**
 `Run` オブジェクト。 



**例外:**
 
 - `Error`:  run の初期化中に未知または内部エラーが発生した場合。 
 - `AuthenticationError`:  ユーザー が有効な認証情報を提供できなかった場合。 
 - `CommError`:  W&B サーバー との通信に問題があった場合。 
 - `UsageError`:  ユーザー が無効な 引数 を指定した場合。 
 - `KeyboardInterrupt`:  ユーザー が run を中断した場合。 



**使用例:**
 `wandb.init()` は `Run` オブジェクトを返します。Run オブジェクトを使ってデータをログし、Artifacts を保存し、run のライフサイクルを管理します。 

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    # run に accuracy と loss をログする
    acc = 0.95  # 精度の例
    loss = 0.05  # 損失の例
    run.log({"accuracy": acc, "loss": loss})
```