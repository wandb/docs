---
title: 設定
object_type: python_sdk_actions
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py >}}

W&B SDK の設定

このクラスは W&B SDK の設定を管理し、すべての設定に型安全性とバリデーションを確保します。設定は属性としてアクセスでき、プログラムコード、環境変数（`WANDB_` プレフィックス付き）、設定ファイルを通して初期化できます。

設定は以下の 3 つのカテゴリに整理されています。
1. パブリック設定: ユーザーが自身のニーズに合わせて安全に変更可能な、W&B の中核的な設定オプション
2. 内部設定: 'x_' で始まる接頭辞つきの設定で、SDK の低レベルな挙動を扱います。
   これらは主に内部用途やデバッグ向けです。変更可能ですが、公開 API の一部ではなく、将来のバージョンで予告なく変更される場合があります。
3. 計算済み設定: 他の設定や環境から自動的に導出される、読み取り専用（Read-only）の設定です。

属性一覧:
- allow_offline_artifacts (bool): テーブルアーティファクトをオフラインモードで同期できるかどうかのフラグ。従来の挙動に戻したい場合は False にしてください。
- allow_val_change (bool): `Config` の値が設定後に変更可能かどうかのフラグ。
- anonymous (Optional): 匿名データロギングの制御。
    選択肢は次の通りです:
    - "never" : W&B アカウントをリンクしないと run のトラッキング不可。匿名 run を誤って作成しないための設定。
    - "allow" : ログイン済みユーザーは自分のアカウントで run をトラッキングでき、未ログイン実行の場合はチャートが UI で確認可能。
    - "must" : ログイン済みユーザーの代わりに匿名アカウントへ run を送信します。
- api_key (Optional): W&B の APIキー。
- azure_account_url_to_access_key (Optional): Azure の連携用にアカウントURLとそのアクセスキーを紐付けるマッピング。
- base_url (str): データ同期先となる W&B バックエンドの URL。
- code_dir (Optional): W&B でトラッキングするコードが格納されているディレクトリー。
- config_paths (Optional): `Config` オブジェクトへ読み込む設定ファイルへのパス一覧。
- console (Literal): 適用するコンソールキャプチャのタイプ。
    選択肢:
    "auto" - システム環境や設定に基づき自動でコンソールキャプチャ方法を選択
    "off" - コンソールキャプチャを無効化
    "redirect" - 出力をキャプチャするためファイルディスクリプタをリダイレクト
    "wrap" - sys.stdout/sys.stderr の write メソッドを上書き。
    "wrap_raw" - "wrap" と同様だが、エミュレータを介さず生の出力を直接キャプチャ
    "wrap_emu" - エミュレータを介して出力をキャプチャ
- console_multipart (bool): マルチパートのコンソールログファイルを生成するかどうか。
- credentials_file (str): 一時的なアクセス トークンを書き込むファイルパス。
- disable_code (bool): コードのキャプチャを無効にするかどうか。
- disable_git (bool): git 状態のキャプチャを無効にするかどうか。
- disable_job_creation (bool): W&B Launch 用のジョブアーティファクト作成可否。
- docker (Optional): スクリプト実行時に使用する Docker イメージ。
- email (Optional): ユーザーのメールアドレス。
- entity (Optional): W&B エンティティ（例: user や team）。
- force (bool): `wandb.login()` に `force` フラグを渡すかどうか。
- fork_from (Optional): 既存 run の実行履歴からブランチ（fork）する際の指定。
    該当する実行箇所は run の ID、メトリクス、その値で定義されます。
    現在サポートされているメトリクスは '_step' のみです。
- git_commit (Optional): run と関連付ける git コミットハッシュ。
- git_remote (str): run と関連付ける git remote。
- git_remote_url (Optional): git リモートリポジトリの URL。
- git_root (Optional): git リポジトリのルートディレクトリー。

- host (Optional): スクリプトを実行しているマシンのホスト名。
- http_proxy (Optional): W&B への http リクエスト用プロキシサーバー。
- https_proxy (Optional): W&B への https リクエスト用プロキシサーバー。
- identity_token_file (Optional): 認証用のアイデンティティトークン（JWT）が記載されたファイルのパス。
- ignore_globs (Sequence): アップロードから除外するファイルを指定する `files_dir` 相対の Unix グロブパターン。
- init_timeout (float): `wandb.init` の完了を待つ最大秒数。
- insecure_disable_ssl (bool): SSL 検証を無効（安全でない）にするかどうか。
- job_name (Optional): 実行する Launch ジョブの名称。
- job_source (Optional): Launch のソースタイプ。
- label_disable (bool): 自動ラベリング機能の無効化可否。

- launch_config_path (Optional): Launch 設定ファイルへのパス。
- login_timeout (Optional): ログイン操作のタイムアウト（秒）。
- mode (Literal): W&B のロギング・同期動作モード。
- notebook_name (Optional): Jupyter 環境などで実行している場合のノートブック名。
- organization (Optional): W&B オーガニゼーション。
- program (Optional): run を作成したスクリプトのパス（存在する場合）。
- program_abspath (Optional): リポジトリのルートから run を作成したスクリプトへの絶対パス。
    ルートディレクトリーは .git ディレクトリーが存在すればその位置。なければカレントワーキングディレクトリー。
- program_relpath (Optional): run を作成したスクリプトへの相対パス。
- project (Optional): W&B Project の ID。
- quiet (bool): 重要ではない出力を抑制するフラグ。
- reinit (Union): アクティブな run がある状態で `wandb.init()` が呼ばれたときの挙動。
    オプション:
    - "default": ノートブックでは "finish_previous"、それ以外は "return_previous"
    - "return_previous": 未終了の最新 run を返す（`wandb.run`は更新されません）。
    - "finish_previous": アクティブ run をすべて終了し新しい run を返す。
    - "create_new": 他のアクティブ run を変更せず、新しい run を作成。
    ※ ブール形式もサポートされていますが非推奨。False は "return_previous"、True は "finish_previous" と同等。
- relogin (bool): 強制的に再ログインを試みるかどうか。
- resume (Optional): run の再開挙動を指定。
    オプション:
    - "must": 同じ ID の既存 run から再開。なければ失敗。
    - "allow": 同じ ID の既存 run からの再開を試み、なければ新規で作成。
    - "never": 常に新規 run を開始。同じ ID の run が既に存在する場合は失敗。
    - "auto": 同一マシン上で最近失敗した run から自動的に再開。
- resume_from (Optional): 既存 run の実行履歴から再開するポイントを指定。
    対象ポイントは run の ID、メトリクスとその値で定義。
    現在サポートされているメトリクスは '_step' のみ。

- root_dir (str): run 関連のパスの基準となるルートディレクトリー。
    特に wandb ディレクトリーや run ディレクトリーの導出に使用されます。
- run_group (Optional): 関連する run をグループ化するための識別子。
    UI 上でのグループ表示に利用。
- run_id (Optional): run の ID。
- run_job_type (Optional): 実行しているジョブの種類（例：トレーニング、評価 など）。
- run_name (Optional): run のわかりやすい名前。
- run_notes (Optional): run の補足説明やメモ。
- run_tags (Optional): run に紐付けるタグ。（整理や絞り込み用）
- sagemaker_disable (bool): SageMaker に特化した機能を無効にするフラグ。
- save_code (Optional): run に紐付くコードを保存するかどうか。
- settings_system (Optional): システム共通の設定ファイルへのパス。

- show_errors (bool): エラーメッセージの表示可否。
- show_info (bool): 情報メッセージの表示可否。
- show_warnings (bool): 警告メッセージの表示可否。
- silent (bool): すべての出力を抑制するフラグ。

- strict (Optional): バリデーションやエラーチェックで厳格モードを有効化するかどうか。
- summary_timeout (int): summary 操作のタイムアウト（秒）。

- sweep_id (Optional): この run が所属する Sweep の識別子。
- sweep_param_path (Optional): sweep パラメータ設定ファイルへのパス。
- symlink (bool): シンボリックリンクを使用するかどうか（Windows 以外ではデフォルトで True）。
- sync_tensorboard (Optional): TensorBoard ログと W&B の同期可否。
- table_raise_on_max_row_limit_exceeded (bool): テーブルの行数制限超過時に例外を投げるかどうか。
- username (Optional): ユーザー名。

- x_skip_transaction_log (bool): run のイベントをトランザクションログに保存しないかどうか。
    オンライン run でのみ有効。ディスクへの書き込み量を削減できます。
    復旧保証がなくなるため注意して使用してください。

- x_stats_open_metrics_endpoints (Optional): システムメトリクス監視用 OpenMetrics の `/metrics` エンドポイント一覧。
- x_stats_open_metrics_filters (Union): OpenMetrics `/metrics` エンドポイントから収集するメトリクスに適用するフィルタ。
    サポートされている形式:
    - {"メトリクスの正規表現パターン（エンドポイント名を接頭辞として含む）": {"label": "ラベル値の正規表現パターン"}}
    - ("メトリクス正規表現パターン1", "メトリクス正規表現パターン2", ...)