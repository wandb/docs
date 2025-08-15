---
title: 'init()

  '
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

W&B で run を開始し、追跡・ログを行うための関数です。

ML トレーニングパイプラインでは、トレーニングスクリプトや評価スクリプトのはじめに `wandb.init()` を追加することで、それぞれが W&B 上で個別の run としてトラッキングされます。

`wandb.init()` を呼ぶと、run 用のログをバックグラウンドプロセスが開始されます。またデフォルトで https://wandb.ai にデータがリアルタイム同期されるため、すぐに結果を確認できます。データのログが終わったら、`wandb.finish()` を呼んで run を終了してください。`run.finish()` を呼ばなくても、スクリプトの終了時に自動的に run も終了します。

run の ID には、次の特殊文字 `/ \ # ? % :` は使用できません。



**引数:**
 
 - `entity`:  この run を記録する Entities（ユーザー名またはチーム名）。Entities は事前に UI で作成する必要があります。指定しない場合はデフォルトの entity になります。デフォルト entity の変更は、設定の「Default location to create new projects」から行えます。
 - `project`:  この run を紐付ける Projects 名。未指定の場合は、git の root や実行中スクリプト名に基づき自動推定されます。推定できなければ `"uncategorized"` が使用されます。
 - `dir`:  実験ログやメタデータファイルを保存する絶対パス。未指定時は `./wandb` ディレクトリーになります。なお、`download()` でアーティファクトを保存する場所はここには影響しません。
 - `id`:  再開時に使われる、この run のユニークな識別子。Projects 内でユニークである必要があり、一度削除した run の ID は再利用できません。run の短い説明名は `name` フィールド、複数 run でハイパーパラメーターを比較する場合は `config` を利用してください。
 - `name`:  UI 上で分かりやすく識別するための、run の短い表示名。デフォルトではランダムな2語が生成されます。チャートやテーブルの凡例上で名前が簡潔になるため、可読性が高くなります。ハイパーパラメータの保存には `config` の利用を推奨します。
 - `notes`:  Git のコミットメッセージのような run の詳細説明。run の目的やセットアップなど、将来参照する手がかりを残す用途でご利用ください。
 - `tags`:  run をラベル付けするためのタグ一覧。UI での整理や「baseline」「production」などの一時的な識別が可能です。追加・削除やフィルタは UI から簡単に行えます。run の再開時、指定したタグはこちらで全部上書きされます。既存タグに新しいタグを追加したいときは、`run = wandb.init()` 後に `run.tags += ("new_tag",)` を利用してください。
 - `config`:  `wandb.config` に値をセットします。ハイパーパラメータやデータ前処理の設定などを保存する、辞書のようなオブジェクトです。config は UI の概要ページに表示され、group・filter・sort に利用できます。キーにピリオド（`.`）は利用できず、値のサイズは10MB未満である必要があります。辞書型、`argparse.Namespace`、`absl.flags.FLAGS` の場合は、その key-value を `wandb.config` に直接ロードします。文字列を指定すると、YAML ファイルのパスと解釈され、設定値がロードされます。
 - `config_exclude_keys`:  `wandb.config` から除外したい特定のキー一覧。
 - `config_include_keys`:  `wandb.config` へ含めたい特定のキー一覧。
 - `allow_val_change`:  config の値を書き換え可能にするかどうかの制御（初回設定以降の変更を許可するか）。デフォルトでは値を上書きしようとすると例外が発生します。トレーニング中に変化する変数（例：learning rate）を追跡したい場合は `wandb.log()` の利用を検討してください。デフォルトではスクリプト実行環境では `False`、Notebook 環境では `True` です。
 - `group`:  複数の run をひとつの大きな Experiment としてグループ化するための名前。クロスバリデーションや異なるテストセットで複数ジョブを実行する場合などに便利です。UI 上でまとめて操作や一括レビューができるようになります。
 - `job_type`:  グループ内で run の種別を示すためのラベル。たとえばグループ内で「train」「eval」などと区別できます。UI 上でフィルタや比較が容易です。
 - `mode`:  run のデータ管理方法を指定します。
    - `"online"`（デフォルト）: ネットワーク接続があれば W&B とライブ同期し、可視化もリアルタイム反映。
    - `"offline"`: オフライン・エアギャップ環境向け。ローカルに保存し、後からアップロード可能です。run フォルダは同期まで保管しておいてください。
    - `"disabled"`: W&B の全機能を無効化し、run のメソッドが何もしなくなります（テスト時に便利）。
 - `force`:  スクリプト実行に W&B ログインを必須とするかどうか。`True` なら、未ログインの場合スクリプトが実行されません。`False`（デフォルト）なら未ログイン時はオフラインモードになります。
 - `anonymous`:  匿名データログの管理レベルを指定します。
    - `"never"`（デフォルト）: W&B アカウント紐付けを必須とします。実験を個人・チームのアカウントと関連付けることで、意図せず匿名 run が作られるのを防ぎます。
    - `"allow"`: ログインユーザーは自分のアカウントで追跡できますが、W&B アカウント未所持でもチャートやデータの閲覧が可能です。
    - `"must"`: ログイン状態でも匿名アカウントで run を記録します。
 - `reinit`:  "reinit" 設定のショートカットです。run がアクティブな状態で `wandb.init()` した場合の挙動を決定します。
 - `resume`:  指定した `id` を使って run を再開する際の挙動を制御します。
    - `"allow"`: 指定した `id` の run が存在すれば途中から再開、なければ新規作成します。
    - `"never"`: 指定した `id` の run があればエラー、なければ新規作成します。
    - `"must"`: 指定した `id` の run があれば途中から再開、なければエラー。
    - `"auto"`: このマシンでクラッシュした run があれば自動的に再開、それ以外は新規作成。
    - `True`: 非推奨。代わりに `"auto"` を使用してください。
    - `False`: 非推奨。何も設定せず（デフォルト）、常に新しい run を開始します。   
    - `resume` を指定した場合、`fork_from` および `resume_from` は併用不可です。未指定時は常に新規 run が始まります。
 - `resume_from`:  過去の run のあるタイミング（`{run_id}?_step={step}`）から再開する場合に指定します。これで、途中のステップ以降に履歴を切り詰めて再開できます。対象の run は同一プロジェクト内にある必要があります。`id` が同時指定された場合は `resume_from` が優先されます。`resume`、`resume_from`、`fork_from` の併用不可。*ベータ機能であり今後変更の可能性があります。*
 - `fork_from`:  過去 run の任意のタイミング（`{id}?_step={step}`）から新規 run をフォークします。指定した run の履歴ステップからロギングを開始する新しい run を作成します。対象の run は同一プロジェクト内にある必要があります。`id` も指定する場合、`fork_from` と同じ値は使えません（同値ならエラー）。`resume`、`resume_from`、`fork_from` の併用不可。*ベータ機能であり今後変更の可能性があります。*
 - `save_code`:  メインのスクリプトやノートブックを W&B に保存し、実験の再現性確保や run 間のコード比較に役立ちます。デフォルトでは無効ですが、設定画面で有効化できます。
 - `tensorboard`:  非推奨。代わりに `sync_tensorboard` をご利用ください。
 - `sync_tensorboard`:  TensorBoard または TensorBoardX の W&B との自動同期を有効化し、イベントファイルを UI 上で確認できます。
 - `monitor_gym`:  OpenAI Gym 利用時に、環境の動画を自動ログします。
 - `settings`:  実行の高度な設定を含む辞書または `wandb.Settings` オブジェクトを指定します。



**例外:**
 
 - `Error`:  run 初期化中に予期せぬ、または内部的なエラーが発生した場合。
 - `AuthenticationError`:  ユーザー認証情報が無効だった場合。
 - `CommError`:  WandB サーバーとの通信に問題が生じた場合。
 - `UsageError`:  不正な引数をユーザーが指定した場合。
 - `KeyboardInterrupt`:  ユーザーが run を手動で停止した場合。



**戻り値:**
 `Run` オブジェクトを返します。





**使用例:**
 `wandb.init()` は `Run` オブジェクトを返します。このオブジェクトを使ってデータのログ、Artifacts の保存、run のライフサイクル管理が可能です。

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    # 正確率と損失を run にログする
    acc = 0.95  # サンプルの正確率
    loss = 0.05  # サンプルの損失値
    run.log({"accuracy": acc, "loss": loss})
```
