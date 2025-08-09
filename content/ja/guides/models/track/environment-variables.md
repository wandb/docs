---
title: 環境変数
description: W&B の環境変数を設定します。
menu:
  default:
    identifier: ja-guides-models-track-environment-variables
    parent: experiments
weight: 9
---

スクリプトを自動化環境で実行する際は、スクリプト実行前や実行中に環境変数を設定して W&B を制御できます。

```bash
# これは秘密情報なのでバージョン管理にコミットしないでください
WANDB_API_KEY=$YOUR_API_KEY
# 名前やメモの設定は任意です
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# wandb/settings ファイルをコミットしない場合のみ必要です
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# クラウドへの同期を行いたくない場合
os.environ["WANDB_MODE"] = "offline"

# Sweep ID を Run オブジェクトや関連クラスに記録
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## オプションの環境変数

これらのオプション環境変数を使って、リモートマシンでの認証などを設定できます。

| 変数名 | 用途 |
| --------------------------- | ---------- |
| `WANDB_ANONYMOUS` | ユーザーがシークレットURLで匿名の run を作成できるように、`allow`、`never`、または `must` を指定します。 |
| `WANDB_API_KEY` | アカウントに紐づく認証キーを設定します。 キーは [設定ページ](https://app.wandb.ai/settings) で確認できます。リモートマシンで `wandb login` を実行していない場合、必須です。 |
| `WANDB_BASE_URL` | [wandb/local]({{< relref path="/guides/hosting/" lang="ja" >}}) を使う場合、この環境変数に `http://YOUR_IP:YOUR_PORT` を設定してください。 |
| `WANDB_CACHE_DIR` | デフォルトは \~/.cache/wandb ですが、この環境変数で保存場所を上書きできます。 |
| `WANDB_CONFIG_DIR` | デフォルトは \~/.config/wandb ですが、この環境変数で場所を上書きできます。|
| `WANDB_CONFIG_PATHS` | 読み込む yaml ファイルのカンマ区切りリスト。 wandb.config にロードされます。詳しくは [config]({{< relref path="./config.md#file-based-configs" lang="ja" >}}) を参照してください。|
| `WANDB_CONSOLE` | "off" にすると stdout / stderr のログを無効化します。対応環境ではデフォルトで "on" です。|
| `WANDB_DATA_DIR` | ステージング artifacts のアップロード先。デフォルトの場所はプラットフォーム依存で、Python パッケージ `platformdirs` の `user_data_dir` の値を使います。このディレクトリーが存在し、実行ユーザーが書き込み権限を持っていることを確認してください。|
| `WANDB_DIR` | 生成ファイルの保存先。未設定の場合、トレーニングスクリプトからの相対 `wandb` ディレクトリになります。このディレクトリーが存在し、書き込み権限があることを確認してください。ダウンロードした artifact の保存先は `WANDB_ARTIFACT_DIR` で個別に設定できます。|
| `WANDB_ARTIFACT_DIR` | ダウンロードした artifact の保存先。未設定の場合、トレーニングスクリプトからの相対 `artifacts` ディレクトリです。このディレクトリーが存在し、書き込み権限があることを確認してください。メタデータファイル生成場所は `WANDB_DIR` で制御できます。|
| `WANDB_DISABLE_GIT` | wandb が git リポジトリの検出やコミット／diff の記録を行わないようにします。|
| `WANDB_DISABLE_CODE` | true にすると wandb がノートブックや git の diff を保存しなくなります。git リポジトリ内の場合は現在のコミットのみ保存します。|
| `WANDB_DOCKER` | docker イメージのダイジェストを指定すると run のリストアが有効になります。これは wandb docker コマンドで自動設定されます。イメージのダイジェストは `wandb docker my/image/name:tag --digest` の実行で取得できます。|
| `WANDB_ENTITY` | run に紐づく entity を指定します。トレーニングスクリプトのディレクトリで `wandb init` を実行すると _wandb_ ディレクトリが作成され、デフォルト entity が保存されます。そのファイルを作成したくない場合や値を上書きしたい場合、この環境変数を使用してください。|
| `WANDB_ERROR_REPORTING` | false に設定すると wandb が致命的エラーをエラートラッキングシステムへ記録しなくなります。|
| `WANDB_HOST` | システムのホスト名ではなく wandb UI 上で表示したいホスト名がある場合に指定します。|
| `WANDB_IGNORE_GLOBS` | 無視したいファイルの glob パターンをカンマ区切りで指定します。これらのファイルはクラウドに同期されません。|
| `WANDB_JOB_NAME` | `wandb` で作成される job の名称を指定します。|
| `WANDB_JOB_TYPE` | "training"や"evaluation"など run のジョブタイプを指定します。詳しくは [grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。|
| `WANDB_MODE` | "offline" にすると run のメタデータをローカルに保存し、サーバー同期を行いません。`disabled` では wandb 機能を完全にオフにします。|
| `WANDB_NAME` | run のわかりやすい名前。未指定の場合は自動生成されます。|
| `WANDB_NOTEBOOK_NAME` | Jupyter 環境で notebook 名をこの変数で指定できます。自動検出も試みます。|
| `WANDB_NOTES` | run についての長めのメモを記入可能。Markdown に対応し、後から UI で編集できます。|
| `WANDB_PROJECT` | run の紐付け先 project。`wandb init` でも設定できますが、環境変数の方が優先されます。|
| `WANDB_RESUME` | デフォルトは _never_。_auto_ で失敗した run を自動再開、_must_ で起動時にrunの存在を必須にします。常に自分でユニークなIDを使いたい場合は _allow_ にして `WANDB_RUN_ID` を常に指定してください。|
| `WANDB_RUN_GROUP` | experiment 名を指定することで自動的に run をグループ化できます。詳細は [grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。|
| `WANDB_RUN_ID` | スクリプト1回実行ごとにプロジェクト内でグローバルに一意な文字列を指定します（最大64文字）。単語以外の文字はダッシュに変換されます。失敗時に既存 run の再開などに利用できます。|
| `WANDB_QUIET` | `true` で、標準出力へのログ出力を重大なもののみに抑えます。この場合、全てのログは `$WANDB_DIR/debug.log` に保存されます。|
| `WANDB_SILENT` | `true` で wandb のログ出力を全て無効にします。スクリプト化されたコマンドで便利です。すべてのログは `$WANDB_DIR/debug.log` に保存されます。|
| `WANDB_SHOW_RUN` | `true` に設定すると run の URL を自動的にブラウザで開きます（OS が対応している場合）。|
| `WANDB_SWEEP_ID` | Sweep ID を `Run` オブジェクトや関連クラスに付与し、UI に表示します。|
| `WANDB_TAGS` | run に適用するタグをカンマ区切りリストで指定します。|
| `WANDB_USERNAME` | チーム内メンバーのユーザー名を指定します。サービスアカウント用 APIキーと併用して自動化 run の帰属先として指定できます。|
| `WANDB_USER_EMAIL` | チーム内メンバーのメールアドレスを指定します。サービスアカウント用 APIキーと併用して自動化 run の帰属先として指定できます。|

## Singularity 環境での利用

[Singularity](https://singularity.lbl.gov/index.html) でコンテナを実行する場合、上記変数の前に `SINGULARITYENV_` を付けて環境変数を渡せます。Singularity の環境変数に関する詳細は [こちら](https://singularity.lbl.gov/docs-environment-metadata#environment) を参照してください。

## AWS での実行

AWS バッチジョブで実行する場合は、W&B 資格情報でマシン認証を簡単に行えます。[設定ページ](https://app.wandb.ai/settings) から API キーを取得し、[AWS バッチジョブ定義](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters)内で `WANDB_API_KEY` 環境変数を設定してください。