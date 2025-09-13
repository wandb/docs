---
title: 環境変数
description: W&B の 環境 変数を設定する。
menu:
  default:
    identifier: ja-guides-models-track-environment-variables
    parent: experiments
weight: 9
---

自動化された環境でスクリプトを実行する場合、スクリプトの実行前またはスクリプト内で設定した環境変数で W&B を制御できます。

```bash
# これは秘密なので、バージョン管理にコミットしないでください
WANDB_API_KEY=$YOUR_API_KEY
# 名前とノートは任意
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# wandb/settings ファイルをチェックインしない場合にのみ必要です
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# スクリプトをクラウドと同期したくない場合
os.environ["WANDB_MODE"] = "offline"

# Sweep ID のトラッキングを Run オブジェクトおよび関連クラスに追加
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## オプションの環境変数

これらのオプションの環境変数を使って、リモートマシンでの認証設定などを行えます。

| Variable name | Usage |
| --------------------------- | ---------- |
| `WANDB_ANONYMOUS` | `allow`、`never`、`must` のいずれかを指定すると、ユーザーが秘密の URL で匿名の Runs を作成できるように制御できます。 |
| `WANDB_API_KEY` | アカウントに紐づく認証キーを設定します。キーは [your settings page](https://app.wandb.ai/settings) で確認できます。リモートマシンで `wandb login` を実行していない場合は必須です。 |
| `WANDB_BASE_URL` | [wandb/local]({{< relref path="/guides/hosting/" lang="ja" >}}) を使っている場合は、この環境変数を `http://YOUR_IP:YOUR_PORT` に設定してください。 |
| `WANDB_CACHE_DIR` | 既定は \~/.cache/wandb です。この環境変数で場所を上書きできます。 |
| `WANDB_CONFIG_DIR` | 既定は \~/.config/wandb です。この環境変数で場所を上書きできます。 |
| `WANDB_CONFIG_PATHS` | wandb.config に読み込む YAML ファイルのカンマ区切りリスト。詳しくは [config]({{< relref path="./config.md#file-based-configs" lang="ja" >}}) を参照してください。 |
| `WANDB_CONSOLE` | "off" に設定すると stdout / stderr のロギングを無効化します。対応する環境では既定は "on" です。 |
| `WANDB_DATA_DIR` | ステージングした Artifacts をアップロードする場所。既定の場所は使用しているプラットフォームに依存し、`platformdirs` Python パッケージの `user_data_dir` の値を使います。ディレクトリーが存在し、実行ユーザーが書き込み可能であることを確認してください。 |
| `WANDB_DIR` | 生成されたすべてのファイルを保存する場所。未設定の場合、トレーニングスクリプトからの相対パスで `wandb` ディレクトリーが既定になります。ディレクトリーが存在し、実行ユーザーが書き込み可能であることを確認してください。これはダウンロードした Artifacts の場所は制御しません。Artifacts の保存先は `WANDB_ARTIFACT_DIR` 環境変数で設定できます。 |
| `WANDB_ARTIFACT_DIR` | ダウンロードしたすべての Artifacts を保存する場所。未設定の場合、トレーニングスクリプトからの相対パスで `artifacts` ディレクトリーが既定になります。ディレクトリーが存在し、実行ユーザーが書き込み可能であることを確認してください。これは生成されたメタデータファイルの場所は制御しません。メタデータの保存先は `WANDB_DIR` 環境変数で設定できます。 |
| `WANDB_DISABLE_GIT` | wandb が Git リポジトリを探索して最新のコミット / diff を取得するのを防ぎます。 |
| `WANDB_DISABLE_CODE` | true に設定すると、wandb がノートブックや Git の diff を保存しないようにします。Git リポジトリ内にいる場合は現在のコミットは引き続き保存されます。 |
| `WANDB_DOCKER` | Runs の復元を有効化するために、Docker イメージのダイジェストを設定します。これは wandb の docker コマンドで自動設定されます。イメージのダイジェストは `wandb docker my/image/name:tag --digest` の実行で取得できます。 |
| `WANDB_ENTITY` | Run に関連付けられた Entity。トレーニングスクリプトのディレクトリーで `wandb init` を実行すると、_wandb_ というディレクトリーが作成され、デフォルトの Entity が保存されます（ソース管理にチェックイン可能）。そのファイルを作成したくない、または値を上書きしたい場合はこの環境変数を使用できます。 |
| `WANDB_ERROR_REPORTING` | false に設定すると、wandb が致命的エラーをエラー追跡システムにログしないようにします。 |
| `WANDB_HOST` | システム提供のホスト名を使いたくない場合に、wandb のインターフェースに表示したいホスト名を設定します。 |
| `WANDB_IGNORE_GLOBS` | 無視するファイルの glob をカンマ区切りで指定します。これらのファイルはクラウドへ同期されません。 |
| `WANDB_JOB_NAME` | `wandb` が作成するあらゆるジョブの名前を指定します。 |
| `WANDB_JOB_TYPE` | "training" や "evaluation" など、Run の種類を示すジョブタイプを指定します。詳しくは [グループ化]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。 |
| `WANDB_MODE` | "offline" に設定すると、wandb は Run のメタデータをローカルに保存し、サーバーに同期しません。`disabled` に設定すると wandb を完全に無効化します。 |
| `WANDB_NAME` | Run のわかりやすい名前。未設定の場合はランダムに生成されます。 |
| `WANDB_NOTEBOOK_NAME` | Jupyter で実行している場合、この変数でノートブック名を設定できます。自動検出も試みます。 |
| `WANDB_NOTES` | Run に関する詳細なノート。Markdown が使用でき、後で UI から編集できます。 |
| `WANDB_PROJECT` | Run に関連付けられた Project。これは `wandb init` でも設定できますが、環境変数が値を上書きします。 |
| `WANDB_RESUME` | 既定では _never_。_auto_ に設定すると、wandb は失敗した Runs を自動で再開します。_must_ に設定すると、起動時に Run の存在を強制します。常に独自の一意な ID を生成したい場合は _allow_ に設定し、常に `WANDB_RUN_ID` を設定してください。 |
| `WANDB_RUN_GROUP` | Runs を自動的にまとめるための Experiment 名を指定します。詳しくは [グループ化]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。 |
| `WANDB_RUN_ID` | スクリプトの単一の Run に対応する、（Project ごとに）グローバルに一意な文字列を設定します。64 文字以内である必要があります。単語文字以外のすべての文字はダッシュに変換されます。失敗時に既存の Run を再開する用途に使えます。 |
| `WANDB_QUIET` | `true` に設定すると、標準出力に出すメッセージを重要なもののみに制限します。設定時はすべてのログが `$WANDB_DIR/debug.log` に書き出されます。 |
| `WANDB_SILENT` | `true` に設定すると、wandb のログメッセージを出力しません。スクリプト化したコマンドに便利です。設定時はすべてのログが `$WANDB_DIR/debug.log` に書き出されます。 |
| `WANDB_SHOW_RUN` | `true` に設定すると、OS が対応している場合に Run の URL を自動的にブラウザーで開きます。 |
| `WANDB_SWEEP_ID` | Run オブジェクトおよび関連クラスに Sweep ID のトラッキングを追加し、UI に表示します。 |
| `WANDB_TAGS` | Run に付与するタグのカンマ区切りリスト。 |
| `WANDB_USERNAME` | Run に関連付けられたチームメンバーのユーザー名。サービスアカウントの API キーと併用して、自動化された Runs をチームメンバーに帰属させることができます。 |
| `WANDB_USER_EMAIL` | Run に関連付けられたチームメンバーのメールアドレス。サービスアカウントの API キーと併用して、自動化された Runs をチームメンバーに帰属させることができます。 |

## Singularity 環境

[Singularity](https://singularity.lbl.gov/index.html) でコンテナを実行する場合は、上記の変数名の前に `SINGULARITYENV_` を付けることで環境変数を渡せます。Singularity の環境変数に関する詳細は[こちら](https://singularity.lbl.gov/docs-environment-metadata#environment)を参照してください。

## AWS での実行

AWS でバッチジョブを実行する場合、W&B の認証情報でマシンを簡単に認証できます。[settings page](https://app.wandb.ai/settings) から API キーを取得し、[AWS batch job spec](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters) で `WANDB_API_KEY` 環境変数を設定してください。