---
title: 設定
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-classes-Settings
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py >}}

W&B SDK の設定

このクラスは、W&B SDK の設定を管理します。すべての設定項目に対して型安全性とバリデーションを保証します。設定値は属性としてアクセスでき、プログラム内で初期化したり、環境変数（`WANDB_` プレフィックス）、設定ファイルからも初期化できます。

設定は3つのカテゴリに整理されています：
1. パブリック設定：ユーザーが自分の用途に合わせて安全に変更可能な基本設定です。W&B の振る舞いをカスタマイズできます。
2. 内部設定：`x_` で始まる設定。SDK の低レベルの振る舞いを制御します。主に内部利用やデバッグ用で、将来のバージョンで予告なく変更されることがあります。
3. 計算済み設定：他の設定や環境から自動的に導出される読み取り専用設定です。

属性一覧：
- allow_offline_artifacts (bool): オフラインモードで Table Artifacts を同期させるかどうかのフラグ。従来の動作に戻したい場合は False にします。
- allow_val_change (bool): 一度設定された `Config` の値を変更できるかどうかのフラグ。
- anonymous (Optional): 匿名データロギングの制御。
    設定値例:
    - "never": W&B アカウントと連携必須。意図せず匿名 run を作ることを防ぎます。
    - "allow": ログイン済みの場合は通常どおりトラッキング、アカウントがない場合は UIでチャート閲覧のみ可能。
    - "must": サインアップ済みユーザーの代わりに匿名アカウントへ run を送信します。
- api_key (Optional): W&B の APIキー。
- azure_account_url_to_access_key (Optional): Azure インテグレーション用、Azure アカウント URL と対応するアクセスキーのマッピング。
- base_url (str): W&B バックエンド（同期先）のURL。
- code_dir (Optional): W&B でトラッキングするコードが入ったディレクトリー。
- config_paths (Optional): 設定ファイルのパスリスト。これらのファイルから `Config` オブジェクトに設定を読み込みます。
- console (Literal): 適用するコンソールキャプチャの方式。
    設定値：
    "auto" - システム環境や設定に基づき自動選択
    "off" - コンソールキャプチャを無効化
    "redirect" - 出力キャプチャのため低レベルなファイルディスクリプターをリダイレクト
    "wrap" - sys.stdout/sys.stderr の write メソッドを上書き。システム状態により "wrap_raw" または "wrap_emu" にマッピングされます。
    "wrap_raw" - "wrap" 同様だがエミュレーターを介さず出力をそのままキャプチャ（手動での指定非推奨）
    "wrap_emu" - "wrap" 同様だがエミュレーターを介して出力をキャプチャ（手動での指定非推奨）
- console_multipart (bool): マルチパート形式のコンソールログファイルを出力するかどうか。
- credentials_file (str): 一時的なアクセストークンを書き出すファイルパス。
- disable_code (bool): コードキャプチャを無効化するかどうか。
- disable_git (bool): git の状態キャプチャを無効化するかどうか。
- disable_job_creation (bool): W&B Launch 向けのジョブアーティファクト作成を無効化。
- docker (Optional): スクリプト実行時に使用する Docker イメージ。
- email (Optional): ユーザーのメールアドレス。
- entity (Optional): W&B の Entity（ユーザーや Team など）。
- force (bool): `wandb.login()` に `force` フラグを渡すかどうか。
- fork_from (Optional): 以前の run の特定タイミング（run ID・メトリクス・値）から fork する場合に指定します。現時点ではメトリクス "_step" のみ対応。
- git_commit (Optional): この run に紐づける git コミットのハッシュ値。
- git_remote (str): この run に紐づける git remote。
- git_remote_url (Optional): git remote リポジトリーの URL。
- git_root (Optional): git リポジトリーのルートディレクトリー。

- host (Optional): スクリプト実行マシンのホスト名。
- http_proxy (Optional): W&B への http リクエスト用カスタムプロキシサーバー。
- https_proxy (Optional): W&B への https リクエスト用カスタムプロキシサーバー。
- identity_token_file (Optional): 認証用のアイデンティティトークン（JWT）を含むファイルのパス。
- ignore_globs (Sequence): `files_dir` 内でアップロード対象外にするファイルの Unix グロブパターン。
- init_timeout (float): `wandb.init` の完了を待つ最大秒数。タイムアウトさせたい場合に指定します。
- insecure_disable_ssl (bool): SSL 検証を安全でなく無効化するかどうか。
- job_name (Optional): スクリプトを実行する Launch ジョブ名。
- job_source (Optional): Launch のソースタイプ。
- label_disable (bool): 自動ラベル機能を無効化するか。

- launch_config_path (Optional): Launch 設定ファイルのパス。
- login_timeout (Optional): ログイン処理をタイムアウトさせるまでの秒数。
- mode (Literal): W&B のロギング・同期動作のモード。
- notebook_name (Optional): Jupyter などノートブック環境でのノートブック名。
- organization (Optional): W&B の Organization 名。
- program (Optional): run を作成したスクリプトのパス。
- program_abspath (Optional): スクリプトの絶対パス（git プロジェクトの root からのパス）。.git ディレクトリーがあればそのディレクトリ、なければカレントディレクトリーを root とみなします。
- program_relpath (Optional): run を作成したスクリプトの相対パス。
- project (Optional): W&B の Project の ID。
- quiet (bool): 重要でない出力を抑制するフラグ。
- reinit (Union): 既存の run がアクティブな状態で `wandb.init()` を呼んだ際の挙動。
    オプション：
    - "default": ノートブックでは "finish_previous"、それ以外では "return_previous"
    - "return_previous": 未終了 run のうち最新を返す（`wandb.run`は更新されません）
    - "finish_previous": すべてのアクティブな run を終了し、新しい run を返す
    - "create_new": 他のアクティブ run に干渉せず新しい run を作成（`wandb.run`や上位APIは更新されません。古いインテグレーションは利用できない場合あり）
    真偽値の指定も可能ですが非推奨です。False は "return_previous"、True は "finish_previous" と同等です。
- relogin (bool): 新しいログインを強制するかどうか。
- resume (Optional): run の再開時の挙動。
    オプション：
    - "must": 同じIDの既存 run からのみ再開。見つからなければ失敗します。
    - "allow": 同じIDの既存 run があれば再開、なければ新規作成。
    - "never": 毎回新規 run として開始。同じIDの run があれば失敗します。
    - "auto": 同じマシンで最近失敗した run から自動再開。
- resume_from (Optional): 以前の run の特定ポイント（run ID・メトリクス名・値）から再開。現時点で "_step" のみサポート。

- root_dir (str): すべての run 関連パスの基準となる root ディレクトリー。主に wandb ディレクトリーや run ディレクトリー算出に使用されます。
- run_group (Optional): 関連する run をグループ化するための識別子。UI でのグループ化に利用。
- run_id (Optional): run のID。
- run_job_type (Optional): 実行しているジョブのタイプ（例：training、evaluation）。
- run_name (Optional): run のわかりやすい名前。
- run_notes (Optional): run に追加する補足説明やノート。
- run_tags (Optional): run に付けるタグ。整理やフィルタリングに利用。
- sagemaker_disable (bool): SageMaker 固有機能を無効化するフラグ。
- save_code (Optional): run に関連付けるコードを保存するかどうか。
- settings_system (Optional): システム全体に適用する設定ファイルのパス。

- show_errors (bool): エラーメッセージを表示するかどうか。
- show_info (bool): 情報メッセージを表示するかどうか。
- show_warnings (bool): 警告メッセージを表示するかどうか。
- silent (bool): すべての出力を抑制するかどうかのフラグ。

- strict (Optional): バリデーションやエラーチェックで厳格モードを有効化するかどうか。
- summary_timeout (int): サマリー操作のタイムアウト時間（秒）。

- sweep_id (Optional): この run が所属する Sweep の識別子。
- sweep_param_path (Optional): Sweep のパラメータ設定ファイルへのパス。
- symlink (bool): シンボリックリンクを利用するか。Windows 以外ではデフォルトで True。
- sync_tensorboard (Optional): TensorBoard のログを W&B と同期するかどうか。
- table_raise_on_max_row_limit_exceeded (bool): テーブルの行数上限超過時に例外を発生させるか。
- username (Optional): ユーザー名。

- x_skip_transaction_log (bool): run のイベントをトランザクションログに保存しないかどうか。オンライン run の時のみ有効です。ディスクへの書き込み量削減に使えますが、リカバリ保証はなくなるため注意してください。

- x_stats_open_metrics_endpoints (Optional): システムメトリクス監視のための OpenMetrics `/metrics` エンドポイント。
- x_stats_open_metrics_filters (Union): OpenMetrics `/metrics` から収集したメトリクスに適用するフィルター。
    2つの形式をサポートします：
    - {"メトリクス正規表現パターン (エンドポイント名含む)": {"label": "ラベル値の正規表現パターン"}}
    - ("メトリクス正規表現パターン1", "メトリクス正規表現パターン2", ...)