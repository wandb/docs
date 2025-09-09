---
title: 設定
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-classes-Settings
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py >}}



W&B SDK の 設定。

このクラスは W&B SDK の 設定 を 管理し、すべての 設定 に 対して 型 安全性 と 検証 を 保証します。 設定 は 属性 として アクセス でき、プログラムからの 初期化、環境 変数（`WANDB_ prefix`）、および 設定 ファイルに よる 初期化 に 対応します。

設定 は 次の 3 つの カテゴリ に 整理されています:
1. Public settings: ユーザー が 自身の ニーズ に 合わせて 安全に 変更 できる 中核的な 設定。W&B の 振る舞い を カスタマイズ できます。
2. Internal settings: 'x_' で 始まる、低レベルな SDK の 振る舞い を 扱う 設定。主に 内部 利用 と デバッグ を 想定しています。変更 は 可能ですが、公開 API の 一部 では なく、将来 の バージョン で 予告 なく 変更 される 場合 があります。
3. Computed settings: 他の 設定 や 環境 から 自動的に 算出 される 読み取り専用 の 設定。

Attributes:
- allow_offline_artifacts (bool): オフライン モード で table artifacts を 同期 できる ように する フラグ。従来 の 振る舞い に 戻すには False に 設定 します。
- allow_val_change (bool): いったん 設定 された `Config` の 値 を 変更 できる ように する フラグ。
- anonymous (Optional): 匿名 データ ログ の 制御。
    取りうる 値:
    - "never": run を 追跡 する 前 に W&B アカウント の リンク を 必須 に し、誤って 匿名 run を 作成 しない ように します。
    - "allow": ログイン 済み ユーザー は 自分の アカウント で run を 追跡 できますが、W&B アカウント なし で スクリプト を 実行 している 人 も UI で チャート を 閲覧 できます。
    - "must": サインアップ 済み の ユーザー アカウント では なく、匿名 アカウント に run を 送信 します。
- api_key (Optional): W&B の API キー。
- azure_account_url_to_access_key (Optional): Azure の アカウント URL と アクセス キー の 対応 マップ（Azure インテグレーション 用）。
- base_url (str): データ 同期 の ための W&B バックエンド の URL。
- code_dir (Optional): W&B が 追跡 する コード を 含む ディレクトリー。
- config_paths (Optional): `Config` オブジェクト に 読み込む 設定 ファイル の パス。
- console (Literal): 適用 する コンソール キャプチャ の 種類。
    取りうる 値:
    "auto" - システム の 環境 と 設定 に 基づいて コンソール キャプチャ 方法 を 自動 選択。
    "off" - コンソール キャプチャ を 無効化。
    "redirect" - 出力 を 取得 する ため 低レベル の ファイル ディスクリプタ を リダイレクト。
    "wrap" - sys.stdout/sys.stderr の write メソッド を オーバーライド。システム 状態 に 応じて "wrap_raw" または "wrap_emu" に マップ されます。
    "wrap_raw" - "wrap" と 同様 ですが、エミュレータ を 介さず 生 の 出力 を 直接 取得。`wrap` 設定 から 派生 し、手動 設定 は 非推奨。
    "wrap_emu" - "wrap" と 同様 ですが、エミュレータ を 通じて 出力 を 取得。`wrap` 設定 から 派生 し、手動 設定 は 非推奨。
- console_multipart (bool): マルチパート 形式 の コンソール ログ ファイル を 生成 する か どうか。
- credentials_file (str): 一時的な アクセス トークン を 書き込む ファイル の パス。
- disable_code (bool): コード の 取得 を 無効化 する か どうか。
- disable_git (bool): git 状態 の 取得 を 無効化 する か どうか。
- disable_job_creation (bool): W&B Launch の job artifact 作成 を 無効化 する か どうか。
- docker (Optional): スクリプト を 実行 する Docker イメージ。
- email (Optional): ユーザー の メール アドレス。
- entity (Optional): W&B の Entity（例: Users や Teams）。
- force (bool): `wandb.login()` に `force` フラグ を 渡す か どうか。
- fork_from (Optional): 以前 の run 実行 の ある 時点 から 分岐 する 場所 を 指定。run ID、メトリクス、および その 値 で 定義 します。現時点 では メトリクス '_step' のみ サポート。
- git_commit (Optional): run に 関連付ける git の コミット ハッシュ。
- git_remote (str): run に 関連付ける git リモート。
- git_remote_url (Optional): git リモート リポジトリ の URL。
- git_root (Optional): git リポジトリ の ルート ディレクトリー。

- host (Optional): スクリプト を 実行 している マシン の ホスト名。
- http_proxy (Optional): W&B への http リクエスト 用 の カスタム プロキシ サーバー。
- https_proxy (Optional): W&B への https リクエスト 用 の カスタム プロキシ サーバー。
- identity_token_file (Optional): 認証 用 の アイデンティティ トークン（JWT）を 含む ファイル への パス。
- ignore_globs (Sequence): アップロード から 除外 する ファイル を 指定 する、`files_dir` 基準 の Unix グロブ パターン。
- init_timeout (float): タイムアウト 前 に `wandb.init` 呼び出し が 完了 する まで 待機 する 秒数。
- insecure_disable_ssl (bool): SSL 検証 を 安全 で ない 形 で 無効化 する か どうか。
- job_name (Optional): スクリプト を 実行 している Launch ジョブ 名。
- job_source (Optional): Launch の ソース 種別。
- label_disable (bool): 自動 ラベリング 機能 を 無効化 する か どうか。

- launch_config_path (Optional): Launch 設定 ファイル への パス。
- login_timeout (Optional): ログイン 操作 の タイムアウト 秒数。
- max_end_of_run_history_metrics (int): run 終了 時 に 表示 する History スパークライン の 最大 件数。
- max_end_of_run_summary_metrics (int): run 終了 時 に 表示 する Summary メトリクス の 最大 件数。
- mode (Literal): W&B の ログ と 同期 の 動作 モード。
- notebook_name (Optional): Jupyter 互換 の 環境 で 実行 している 場合 の ノートブック 名。
- organization (Optional): W&B の 組織。
- program (Optional): 可能 な 場合、run を 作成 した スクリプト の パス。
- program_abspath (Optional): run を 作成 した スクリプト への、リポジトリ ルート ディレクトリー からの 絶対 パス。
    リポジトリ の ルート ディレクトリー は、.git ディレクトリー を 含む ディレクトリー が あれば それ を、なければ 現在 の 作業 ディレクトリー を 指します。
- program_relpath (Optional): run を 作成 した スクリプト への 相対 パス。
- project (Optional): W&B の Project ID。
- quiet (bool): 本質的 で ない 出力 を 抑制 する フラグ。
- reinit (Union): run が アクティブ な 間 に `wandb.init()` が 呼ばれた とき の 振る舞い。
    Options:
    - "default": ノートブック では "finish_previous"、それ以外 では "return_previous" を 使用。
    - "return_previous": まだ 終了 していない 最新 の run を 返します。`wandb.run` は 更新 されません。"create_new" オプション を 参照。
    - "finish_previous": すべて の アクティブ な run を 終了 してから、新しい run を 返します。
    - "create_new": 他 の アクティブ な run を 変更 せず 新しい run を 作成。`wandb.run` や `wandb.log` など の トップレベル 関数 は 更新 されません。このため、グローバル run に 依存 する 旧来 の 一部 の インテグレーション は 動作 しません。
    なお、bool も 指定 可能 ですが 非推奨。False は "return_previous"、True は "finish_previous" と 同義 です。
- relogin (bool): 新たな ログイン を 強制 する フラグ。
- resume (Optional): run の レジューム 振る舞い。
    Options:
    - "must": 同じ ID の 既存 の run から レジューム。存在 しない 場合 は 失敗。
    - "allow": 同じ ID の 既存 の run が あれば レジューム。なければ 新規 run を 作成。
    - "never": 常に 新規 run を 開始。同じ ID の run が すでに ある 場合 は 失敗。
    - "auto": 同一 マシン 上 の 直近 の 失敗 した run から 自動 レジューム。
- resume_from (Optional): 以前 の run 実行 の ある 時点 から レジューム する 場所 を 指定。run ID、メトリクス、および その 値 で 定義。現時点 では メトリクス '_step' のみ サポート。

- root_dir (str): すべて の run 関連 パス の 基準 として 使用 する ルート ディレクトリー。
    特に、wandb ディレクトリー と run ディレクトリー の 算出 に 使用 されます。
- run_group (Optional): 関連 する run の グループ 識別子。
    UI 上 で の グルーピング に 使用 されます。
- run_id (Optional): run の ID。
- run_job_type (Optional): 実行 中 の ジョブ の 種別（例: トレーニング、評価）。
- run_name (Optional): 人間 が 読みやすい run 名。
- run_notes (Optional): run に 関する 追加 ノート / 説明。
- run_tags (Optional): 整理 や フィルタリング の ため に run に 付与 する タグ。
- sagemaker_disable (bool): SageMaker 固有 機能 を 無効化 する フラグ。
- save_code (Optional): run に 関連する コード を 保存 する か どうか。
- settings_system (Optional): システム 全体 の 設定 ファイル への パス。


- show_errors (bool): エラー メッセージ を 表示 する か どうか。
- show_info (bool): 情報 メッセージ を 表示 する か どうか。
- show_warnings (bool): 警告 メッセージ を 表示 する か どうか。
- silent (bool): すべて の 出力 を 抑制 する フラグ。

- strict (Optional): 検証 と エラー チェック の 厳格 モード を 有効化 する か どうか。
- summary_timeout (int): Summary 処理 の タイムアウト 秒数。

- sweep_id (Optional): この run が 属する sweep の 識別子。
- sweep_param_path (Optional): sweep パラメータ の 設定 への パス。
- symlink (bool): symlink を 使用 する か どうか（Windows 以外 では 既定 で True）。
- sync_tensorboard (Optional): TensorBoard の ログ を W&B と 同期 する か どうか。
- table_raise_on_max_row_limit_exceeded (bool): テーブル の 行 制限 を 超えた とき に 例外 を 送出 する か どうか。
- username (Optional): ユーザー名。


- x_disable_meta (bool): システム メタデータ の 収集 を 無効化 する フラグ。
- x_disable_stats (bool): システム メトリクス の 収集 を 無効化 する フラグ。


- x_extra_http_headers (Optional): 送信 する すべて の HTTP リクエスト に 追加 する ヘッダー。






















- x_label (Optional): run の ため に 収集 された システム メトリクス と コンソール ログ に 付与 する ラベル。
    フロントエンド で の グルーピング に 使用 され、分散 トレーニング ジョブ における 異なる プロセス の データ を 識別 する の に 役立ちます。




- x_primary (bool): 内部の wandb ファイル と メタデータ を 保存 する か どうか。
    分散 環境 では、メイン の ログ を 担当 する プロセス 以外 では システム メトリクス と ログ のみ が 必要 な 場合 に、ファイル の 上書き を 回避 する の に 有用 です。


- x_save_requirements (bool): requirements ファイル を 保存 する フラグ。
- x_server_side_derived_summary (bool): History から Summary を 自動 計算 する 処理 を サーバー に 委譲 する フラグ。
    これは ユーザー による Summary の 更新 を 無効化 する もの では ありません。


- x_service_wait (float): wandb-core 内部 サービス の 起動 を 待機 する 秒数。
- x_skip_transaction_log (bool): トランザクション ログ への run イベント の 保存 を スキップ する か どうか。
    これは オンライン run にのみ 関係 します。ディスク に 書き込む データ 量 を 減らす ため に 使用 できます。
    ただし、復元 性能 に 関する 保証 が なくなる ため 注意 が 必要 です。




- x_stats_cpu_count (Optional): システム の CPU コア 数。
    設定 すると、run メタデータ の 自動 検出 値 を 上書き します。
- x_stats_cpu_logical_count (Optional): 論理 CPU 数。
    設定 すると、run メタデータ の 自動 検出 値 を 上書き します。

- x_stats_disk_paths (Optional): ディスク 使用量 を 監視 する システム パス。
- x_stats_gpu_count (Optional): GPU デバイス 数。
    設定 すると、run メタデータ の 自動 検出 値 を 上書き します。
- x_stats_gpu_device_ids (Optional): 監視 対象 の GPU デバイス インデックス。
    未設定 の 場合、システム モニタ は すべて の GPU の メトリクス を 取得 します。
    CUDA/ROCm の デバイス 列挙 に 合わせた 0 始まり の インデックス を 想定。
- x_stats_gpu_type (Optional): GPU デバイス の 種類。
    設定 すると、run メタデータ の 自動 検出 値 を 上書き します。

- x_stats_open_metrics_endpoints (Optional): システム メトリクス 取得 用 の OpenMetrics `/metrics` エンドポイント。
- x_stats_open_metrics_filters (Union): OpenMetrics `/metrics` エンドポイント から 収集 した メトリクス に 適用 する フィルター。
    2 つ の 形式 を サポート:
    - {"metric regex pattern, including endpoint name as prefix": {"label": "label value regex pattern"}}
    - ("metric regex pattern 1", "metric regex pattern 2", ...)
- x_stats_open_metrics_http_headers (Optional): OpenMetrics への リクエスト に 追加 する HTTP ヘッダー。

- x_stats_sampling_interval (float): システム モニタ の サンプリング 間隔（秒）。
- x_stats_track_process_tree (bool): `x_stats_pid` を 起点 に、プロセス ツリー 全体 の リソース 使用量 を 監視。
    `True` の 場合、`x_stats_pid` の プロセス と その すべて の 子孫 プロセス の RSS、CPU%、スレッド 数 を 集計 します。
    オーバーヘッド が 発生 する 可能性 が ある ため、既定 では 無効 です。

- x_update_finish_state (bool): この プロセス が サーバー 上 の run の 最終 状態 を 更新 できる か を 示す フラグ。
    分散 トレーニング では、最終 状態 を メイン プロセス のみ が 決定 すべき 場合 に False に 設定 します。