---
title: 環境変数
description: W&B の環境変数を設定します。
menu:
  default:
    identifier: environment-variables
    parent: experiments
weight: 9
---

スクリプトを自動化された環境で実行する際、スクリプト実行前やスクリプト内で環境変数を設定することで W&B を制御できます。

```bash
# これは秘密情報なのでバージョン管理には含めないでください
WANDB_API_KEY=$YOUR_API_KEY
# 名前やノートは任意です
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# wandb/settings ファイルを管理していない場合のみ必要です
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# スクリプトをクラウドと同期したくない場合
os.environ["WANDB_MODE"] = "offline"

# Sweep ID を Run オブジェクトや関連クラスに記録する場合
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## オプションの環境変数

これらのオプション環境変数を使うことで、リモートマシンでの認証などを設定できます。

| 変数名 | 用途 |
| --------------------------- | ---------- |
| `WANDB_ANONYMOUS` | ユーザーがシークレット URL で匿名 run を作成できるように `allow`、`never`、`must` のいずれかを設定します。 |
| `WANDB_API_KEY` | アカウントに紐づく認証キーを指定します。キーは [設定ページ](https://app.wandb.ai/settings) で取得できます。この値はリモートマシンで `wandb login` を実行していない場合に必須です。      |
| `WANDB_BASE_URL` | [wandb/local]({{< relref "/guides/hosting/" >}}) を利用する場合、この環境変数に `http://YOUR_IP:YOUR_PORT` を設定してください。|
| `WANDB_CACHE_DIR` | デフォルトは \~/.cache/wandb です。この環境変数で保存先を変更できます。|
| `WANDB_CONFIG_DIR` | デフォルトは \~/.config/wandb です。この環境変数で保存先を変更できます。|
| `WANDB_CONFIG_PATHS` | 設定ファイル (yaml) のパスをカンマ区切りで指定します。詳細は [config]({{< relref "./config.md#file-based-configs" >}}) を参照してください。|
| `WANDB_CONSOLE` | "off" に設定すると stdout / stderr のログ出力を無効化します。対応環境のデフォルトは "on" です。|
| `WANDB_DATA_DIR` | ステージングアーティファクトをアップロードする場所です。デフォルトの場所はプラットフォームによって異なり、`platformdirs` Python パッケージの `user_data_dir` の値が使われます。このディレクトリーが存在し、実行ユーザーが書込み権限を持っていることを確認してください。|
| `WANDB_DIR` | 生成されたファイルの保存先ディレクトリーです。未設定の場合はトレーニングスクリプトのカレントディレクトリー配下の `wandb` ディレクトリーが使われます。このディレクトリーが存在し、実行ユーザーが書込み権限を持っていることを確認してください。ダウンロードしたアーティファクトの保存先は、`WANDB_ARTIFACT_DIR` 環境変数で制御できます。|
| `WANDB_ARTIFACT_DIR` | ダウンロードしたアーティファクトの保存先ディレクトリーです。未設定の場合はトレーニングスクリプト配下の `artifacts` ディレクトリーが使われます。このディレクトリーが存在し、実行ユーザーが書込み権限を持っていることを確認してください。生成されたメタデータファイルの保存先は、`WANDB_DIR` 環境変数で設定できます。|
| `WANDB_DISABLE_GIT` | git リポジトリの検出や最新コミット/差分の取得を無効化します。|
| `WANDB_DISABLE_CODE` | true に設定すると、notebook や git 差分の保存を無効化します。ただし、git リポジトリ内にいる場合は現在のコミット情報は保存されます。|
| `WANDB_DOCKER` | docker イメージのダイジェストを指定して run の復元を有効にします。この値は wandb docker コマンドで自動設定されます。イメージダイジェストは `wandb docker my/image/name:tag --digest` で取得できます。|
| `WANDB_ENTITY` | 実行する run に紐づける entity を指定します。トレーニングスクリプトのディレクトリーで `wandb init` を実行すると _wandb_ ディレクトリーが作成され、デフォルト entity が保存されます。ファイルを作成したくない場合や上書きしたい時は、この環境変数で指定できます。|
| `WANDB_ERROR_REPORTING` | false に設定すると、wandb による重大エラーのロギングを無効化します。|
| `WANDB_HOST` | システム提供のホスト名ではなく、W&B インターフェースで表示させたい任意のホスト名を設定します。|
| `WANDB_IGNORE_GLOBS` | 無視したいファイルグロブの一覧をカンマ区切りで指定します。これらのファイルはクラウドに同期されません。|
| `WANDB_JOB_NAME` | `wandb` によって作成されるジョブの名前を指定します。|
| `WANDB_JOB_TYPE` | "training" や "evaluation" など run のタイプを指定します。詳細は [grouping]({{< relref "/guides/models/track/runs/grouping.md" >}}) をご覧ください。|
| `WANDB_MODE` | "offline" に設定すると run のメタデータがローカル保存され、サーバーへ同期されません。`disabled` にすると wandb を完全に無効化します。|
| `WANDB_NAME` | run のわかりやすい名前を指定します。未設定時はランダムに生成されます。|
| `WANDB_NOTEBOOK_NAME` | Jupyter 実行時にノートブックの名前をこの変数で設定できます。自動検出も試みられます。|
| `WANDB_NOTES` | run に関する詳細ノートを入力します。Markdown が利用でき、後から UI 上で編集可能です。|
| `WANDB_PROJECT` | run に紐づくプロジェクトを指定します。`wandb init` でも指定できますが、環境変数の値が優先されます。|
| `WANDB_RESUME` | デフォルトは _never_ です。_auto_ にすると失敗した run を自動的に再開します。_must_ にすると、run の開始時に既存 run の存在を強制します。常に独自の一意 ID を生成したい場合は _allow_ にし、`WANDB_RUN_ID` を必ず設定してください。|
| `WANDB_RUN_GROUP` | 実験名を指定し、run を自動的にグループ化します。詳細は [grouping]({{< relref "/guides/models/track/runs/grouping.md" >}}) をご覧ください。|
| `WANDB_RUN_ID` | スクリプトの各 run に対してプロジェクト内でグローバルに一意な文字列 (最大64文字) を設定します。英数字以外の文字はすべてハイフンに変換されます。失敗時の run 再開時などにこの値が利用できます。|
| `WANDB_QUIET` | `true` に設定すると標準出力への出力は重要メッセージのみに制限されます。すべてのログは `$WANDB_DIR/debug.log` に書き込まれます。|
| `WANDB_SILENT` | `true` に設定すると wandb のログステートメントをすべて非表示にします。スクリプト実行コマンドで便利です。すべてのログは `$WANDB_DIR/debug.log` に書き込まれます。|
| `WANDB_SHOW_RUN` | `true` に設定すると、run の URL を自動でブラウザで開きます (OS が対応している場合)。|
| `WANDB_SWEEP_ID` | `Run` オブジェクトや関連クラスに sweep ID を記録し、UI で表示します。|
| `WANDB_TAGS` | run に適用するタグのリストをカンマ区切りで指定します。|
| `WANDB_USERNAME` | run に参加するチームメンバーのユーザー名です。サービスアカウントの API キーと組み合わせて、自動 run の責任者を紐付けられます。|
| `WANDB_USER_EMAIL` | run に参加するチームメンバーのメールアドレスです。サービスアカウントの API キーと組み合わせて、自動 run の責任者を紐付けられます。|

## Singularity 環境

[Singularity](https://singularity.lbl.gov/index.html) コンテナ内で実行する場合、これらの環境変数の前に `SINGULARITYENV_` を付加することで渡すことができます。Singularity の環境変数に関する詳細は [こちら](https://singularity.lbl.gov/docs-environment-metadata#environment) をご参照ください。

## AWS 上での実行

AWS でバッチジョブを実行する場合、W&B の認証情報を用いて簡単に認証できます。[設定ページ](https://app.wandb.ai/settings) から API キーを取得し、[AWS バッチジョブ定義](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters) の環境変数として `WANDB_API_KEY` を設定してください。