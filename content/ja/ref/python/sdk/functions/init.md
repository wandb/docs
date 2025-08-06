---
title: 'init()

  '
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py >}}




### <kbd>function</kbd> `init`

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
    mode: "Literal['online', 'offline', 'disabled'] | None" = None,
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

新しい run を開始し、W&B でトラッキングおよびログを作成します。

ML トレーニングパイプラインでは、トレーニングスクリプトや評価スクリプトの冒頭に `wandb.init()` を追加することで、それぞれが W&B 上の個別の run として記録されます。

`wandb.init()` はバックグラウンドプロセスを生成し、run へのデータのログや https://wandb.ai へのデータの自動同期を行います（デフォルト設定）。そのため、結果をリアルタイムで確認できます。ログの記録を終えたら `wandb.finish()` を呼び出して run を終了してください。もし `run.finish()` を呼ばなかった場合、スクリプト終了時に自動で run も終了します。

Run ID には、次の特殊文字 `/ \ # ? % :` は使用できません。



**引数:**
 
 - `entity`:  この run を記録する username または team name。 entity は事前に存在している必要があるため、UI でアカウントやチームを作成してからログを記録してください。未指定の場合はデフォルト entity となります。デフォルトの entity を変更するには、設定画面の「Default location to create new projects」の「Default team」を更新してください。
 - `project`:  この run を記録する Project の名前。未指定の場合、git のルートや実行中のファイル名などシステム情報に基づいて自動推測します。推測できない場合は `"uncategorized"` が使用されます。
 - `dir`:  実験のログやメタデータファイルを保存するディレクトリの絶対パス。省略時は `./wandb` ディレクトリが使用されます。`download()` で Artifacts を保存する場所には影響しません。
 - `id`:  再開時に使用する、この run のユニークな識別子。Project 内で一意である必要があり、一度削除された後の再利用はできません。短い説明名は `name` を、ハイパーパラメーターの保存には `config` を利用してください。
 - `name`:  この run に付与される短い表示名。UI 上で識別しやすくなります。デフォルトではランダムな2単語の名前が自動生成され、表やチャートから run を素早く参照するのに便利です。簡潔な run 名はチャートやテーブルの表示性も向上します。ハイパーパラメーターの保存は `config` に記入することを推奨します。
 - `notes`:  Git のコミットメッセージのように、この run の詳細な説明を残せます。run の目的やセットアップなど、将来記録を見返す際の参考情報を記入してください。
 - `tags`:  この run にラベルを付けるためのタグのリスト。run の整理や「baseline」「production」などの一時的な識別に有効です。タグは UI から簡単に追加・削除・フィルタできます。run の再開時は、ここで指定したタグが既存のタグを上書きします。既存タグを保持したまま追加するには、`run = wandb.init()` の後で `run.tags += ("new_tag",)` のようにしてください。
 - `config`:  `wandb.config` を設定します。これはモデルのハイパーパラメーターやデータ前処理設定など run の入力パラメータを格納できる辞書型オブジェクトです。UI の概要ページでパラメータごとに run をグルーピング・フィルタ・並び替え可能。キーに `.`（ドット）は使えません。また値は 10MB 未満である必要があります。辞書型、`argparse.Namespace`、`absl.flags.FLAGS` などを渡せばペアがそのまま `wandb.config` に登録されます。文字列を渡す場合は、YAML ファイルパスと解釈され、その内容が config に読み込まれます。
 - `config_exclude_keys`:  `wandb.config` から除外したいキーのリスト。
 - `config_include_keys`:  `wandb.config` に含めたいキーのリスト。
 - `allow_val_change`:  config 値の初期セット後の変更を許可するか制御します。デフォルトでは再代入で例外が発生します。トレーニング中に変化する値（例: 学習率）の記録には、`wandb.log()` の利用を検討してください。スクリプトではデフォルト `False`、ノートブック環境では `True` です。
 - `group`:  複数の run をまとめて一つの実験として整理するグループ名を指定します。クロスバリデーションや複数のテストセットに対してモデルを評価したい場合などに有効です。UI 上でも関連 run をまとめて扱えるため、実験全体の結果を一括確認しやすくなります。
 - `job_type`:  run の種類を指定します。特にグループ内で run を整理したいときに便利です。たとえば "train" や "eval" のような区分を設定し、同種の run どうしを絞り込みや比較がしやすくなります。
 - `mode`:  run データの管理方式を次から選べます:  
    - `"online"`（デフォルト）: W&B とのライブ同期を有効にし、ネットワーク接続時は可視化もリアルタイムで更新されます。
    - `"offline"`: オフラインやセキュアな環境向け。データはローカル保存後、あとから同期できます。同期するには run フォルダーを消さず保持してください。
    - `"disabled"`: W&B の全機能を無効にして、メソッドが何もしなくなります。主にテスト用途で W&B 操作を除外したい場合に利用されます。
 - `force`:  スクリプト実行時に W&B ログインが必須かどうか。`True` なら W&B へログインしていない場合処理を止めます。デフォルトの `False` だと未ログイン時はオフラインモードで続行します。
 - `anonymous`:  匿名データ記録の許可レベルを指定します。  
    - `"never"`（デフォルト）: アカウントと run の紐付けを強制します。意図しない匿名 run の作成を防ぎます。
    - `"allow"`: ログインユーザーが自身のアカウントで run をトラッキングできますが、未ログインで実行した場合も UIでチャートやデータの閲覧が可能です。
    - `"must"`: ログイン状態でも run を必ず匿名アカウントに記録します。
 - `reinit`:  "reinit" 設定の短縮形式。既存の run がアクティブな場合の `wandb.init()` の振る舞いを制御します。
 - `resume`:  特定の `id` を指定した run を再開する際の挙動を制御します。  
    - `"allow"`: 指定した `id` の run が存在すれば前回の step から再開、なければ新規作成。
    - `"never"`: 既にその `id` の run があればエラー。なければ新規作成。
    - `"must"`: 指定した run があれば前回の step から再開、なければエラー。
    - `"auto"`: このマシン上でクラッシュした run があれば自動再開、なければ新規作成。
    - `True`: 非推奨。代わりに `"auto"` を使用してください。
    - `False`: 非推奨。デフォルト（`resume` 未設定）で常に新規作成となります。  
  `resume` を指定した場合、`fork_from` と `resume_from` は同時使用不可です。`resume` を未設定の場合、常に新しい run になります。
 - `resume_from`:  過去の run の特定の時点 `{run_id}?_step={step}` から再開したい場合に指定します。これで履歴を途中で区切って、任意の step から再開が可能です。対象の run は同一 Project 内にある必要があります。`id` も同時指定した場合は `resume_from` が優先されます。`resume`、`resume_from`、`fork_from` は同時利用できません。ベータ機能のため仕様変更の可能性があります。
 - `fork_from`:  過去の run の任意の時点 `{id}?_step={step}` から新しい run を派生させる場合に指定します。対象は同じ Project の run である必要があります。`id` も指定する場合、`fork_from` とは別でなければなりません（同じだとエラー）。`resume`、`resume_from`、`fork_from` の同時利用はできません。ベータ機能のため仕様変更の可能性があります。
 - `save_code`:  メインのスクリプトやノートブックを W&B に保存します。再現性の確保や run ごとのコード比較に役立ちます。デフォルトは無効ですが、設定画面から有効化できます。
 - `tensorboard`:  非推奨。代わりに `sync_tensorboard` を使ってください。
 - `sync_tensorboard`:  TensorBoard や TensorBoardX から W&B へのログ同期を自動化し、関連イベントファイルを W&B UI で閲覧できるよう保存します。
 - `saving relevant event files for viewing in the W&B UI. (Default`:  `False`)
 - `monitor_gym`:  OpenAI Gym を利用時、環境の動画を自動で記録・ログします。
 - `settings`:  上級者向け設定を追加するための辞書または `wandb.Settings` オブジェクト。
 


**例外:**
 
 - `Error`:  run 初期化時に不明または内部エラーが発生した場合。
 - `AuthenticationError`:  適切な認証情報が提供されなかった場合。
 - `CommError`:  WandB サーバーとの通信で問題が発生した場合。
 - `UsageError`:  無効な引数が指定された場合。
 - `KeyboardInterrupt`:  ユーザーによる run の中断。



**戻り値:**
 `Run` オブジェクトが返されます。





**使用例:**
 `wandb.init()` は `Run` オブジェクトを返します。このオブジェクトを利用してデータのログ、Artifacts の保存、run のライフサイクル管理ができます。

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    # 正確度と損失を run にログします
    acc = 0.95  # 例: 正確度
    loss = 0.05  # 例: 損失
    run.log({"accuracy": acc, "loss": loss})
```