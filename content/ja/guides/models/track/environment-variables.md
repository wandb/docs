---
title: Environment variables
description: W&B の 環境 変数 を設定します。
menu:
  default:
    identifier: ja-guides-models-track-environment-variables
    parent: experiments
weight: 9
---

自動化された環境でスクリプトを実行する場合、スクリプトの実行前またはスクリプト内で設定された環境変数で **wandb** を制御できます。

```bash
# これは秘密であり、バージョン管理にチェックインすべきではありません
WANDB_API_KEY=$YOUR_API_KEY
# 名前とメモはオプション
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# wandb/settings ファイルをチェックインしない場合にのみ必要
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# スクリプトをクラウドに同期させたくない場合
os.environ["WANDB_MODE"] = "offline"

# sweep ID トラッキングを Run オブジェクトと関連クラスに追加
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## オプションの環境変数

これらのオプションの環境変数を使用して、リモートマシンでの認証の設定などを行います。

| 変数名 | 使用法 |
| --------------------------- | ---------- |
| **WANDB_ANONYMOUS** | これを `allow`、`never`、または `must` に設定して、ユーザーが秘密の URL で匿名の run を作成できるようにします。 |
| **WANDB_API_KEY** | アカウントに関連付けられた認証 key を設定します。[設定ページ](https://app.wandb.ai/settings) で key を確認できます。リモートマシンで `wandb login` が実行されていない場合は、これを設定する必要があります。 |
| **WANDB_BASE_URL** | [wandb/local]({{< relref path="/guides/hosting/" lang="ja" >}}) を使用している場合は、この環境変数を `http://YOUR_IP:YOUR_PORT` に設定する必要があります |
| **WANDB_CACHE_DIR** | これはデフォルトで ~/.cache/wandb になっています。この場所をこの環境変数で上書きできます |
| **WANDB_CONFIG_DIR** | これはデフォルトで ~/.config/wandb になっています。この場所をこの環境変数で上書きできます |
| **WANDB_CONFIG_PATHS** | wandb.config にロードする yaml ファイルのコンマ区切りリスト。[config]({{< relref path="./config.md#file-based-configs" lang="ja" >}}) を参照してください。 |
| **WANDB_CONSOLE** | stdout / stderr ログを無効にするには、これを "off" に設定します。これは、サポートする環境ではデフォルトで "on" になっています。 |
| **WANDB_DATA_DIR** | ステージング Artifacts がアップロードされる場所。デフォルトの場所はプラットフォームによって異なります。これは、`platformdirs` Python パッケージの `user_data_dir` の値を使用するためです。 |
| **WANDB_DIR** | トレーニングスクリプトからの _wandb_ ディレクトリーの相対位置ではなく、生成されたすべてのファイルをここに保存するには、これを絶対パスに設定します。_このディレクトリーが存在し、プロセスを実行するユーザーが書き込むことができることを確認してください_。これは、ダウンロードされた Artifacts の場所には影響しないことに注意してください。代わりに _WANDB_ARTIFACT_DIR_ を使用して設定できます |
| **WANDB_ARTIFACT_DIR** | トレーニングスクリプトからの _artifacts_ ディレクトリーの相対位置ではなく、ダウンロードされたすべての Artifacts をここに保存するには、これを絶対パスに設定します。このディレクトリーが存在し、プロセスを実行するユーザーが書き込むことができることを確認してください。これは、生成されたメタデータファイルの場所には影響しないことに注意してください。代わりに _WANDB_DIR_ を使用して設定できます |
| **WANDB_DISABLE_GIT** | wandb が git リポジトリをプローブして最新のコミット/差分をキャプチャするのを防ぎます。 |
| **WANDB_DISABLE_CODE** | wandb が ノートブック または git の差分を保存しないようにするには、これを true に設定します。git リポジトリにいる場合は、現在のコミットを保存します。 |
| **WANDB_DOCKER** | run の復元を有効にするには、これを docker イメージのダイジェストに設定します。これは、wandb docker コマンドで自動的に設定されます。`wandb docker my/image/name:tag --digest` を実行して、イメージのダイジェストを取得できます |
| **WANDB_ENTITY** | run に関連付けられたエンティティ。トレーニングスクリプトのディレクトリーで `wandb init` を実行した場合、_wandb_ という名前のディレクトリーが作成され、ソース管理にチェックインできるデフォルトのエンティティが保存されます。そのファイルを作成したくない場合、またはファイルを上書きしたい場合は、環境変数を使用できます。 |
| **WANDB_ERROR_REPORTING** | wandb が致命的なエラーをそのエラー追跡システムにログ記録しないようにするには、これを false に設定します。 |
| **WANDB_HOST** | システムが提供するホスト名を使用したくない場合に、wandb インターフェイスに表示するホスト名を設定します |
| **WANDB_IGNORE_GLOBS** | 無視するファイル glob のコンマ区切りリストにこれを設定します。これらのファイルはクラウドに同期されません。 |
| **WANDB_JOB_NAME** | `wandb` によって作成されたジョブの名前を指定します。 |
| **WANDB_JOB_TYPE** | run のさまざまなタイプを示すために、"トレーニング" や "評価" などのジョブタイプを指定します。詳細については、[グループ化]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。 |
| **WANDB_MODE** | これを "offline" に設定すると、wandb は run のメタデータをローカルに保存し、サーバーに同期しません。これを `disabled` に設定すると、wandb は完全にオフになります。 |
| **WANDB_NAME** | run の人間が読める名前。設定されていない場合は、ランダムに生成されます |
| **WANDB_NOTEBOOK_NAME** | jupyter で実行している場合は、この変数で ノートブック の名前を設定できます。これを自動的に検出しようとします。 |
| **WANDB_NOTES** | run に関するより長いメモ。マークダウンは許可されており、UI で後で編集できます。 |
| **WANDB_PROJECT** | run に関連付けられた プロジェクト。これは `wandb init` で設定することもできますが、環境変数が値を上書きします。 |
| **WANDB_RESUME** | デフォルトでは、これは _never_ に設定されています。_auto_ に設定すると、wandb は失敗した run を自動的に再開します。_must_ に設定すると、起動時に run が強制的に存在します。常に独自のユニークな ID を生成する場合は、これを _allow_ に設定し、常に **WANDB_RUN_ID** を設定します。 |
| **WANDB_RUN_GROUP** | run を自動的にグループ化するための実験名を指定します。詳細については、[グループ化]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。 |
| **WANDB_RUN_ID** | スクリプトの単一の run に対応するグローバルに一意の文字列（プロジェクトごと）にこれを設定します。64 文字以下である必要があります。単語以外の文字はすべてダッシュに変換されます。これは、障害が発生した場合に既存の run を再開するために使用できます。 |
| **WANDB_SILENT** | wandb ログステートメントを非表示にするには、これを **true** に設定します。これが設定されている場合、すべてのログは **WANDB_DIR**/debug.log に書き込まれます |
| **WANDB_SHOW_RUN** | オペレーティングシステムがサポートしている場合、run URL でブラウザーを自動的に開くには、これを **true** に設定します。 |
| **WANDB_SWEEP_ID** | sweep ID トラッキングを `Run` オブジェクトと関連クラスに追加し、UI に表示します。 |
| **WANDB_TAGS** | run に適用されるタグのコンマ区切りリスト。 |
| **WANDB_USERNAME** | run に関連付けられた チーム のメンバーの ユーザー 名。これは、サービスアカウント API key とともに使用して、自動 run の チーム のメンバーへの属性を有効にすることができます。 |
| **WANDB_USER_EMAIL** | run に関連付けられた チーム のメンバーのメール。これは、サービスアカウント API key とともに使用して、自動 run の チーム のメンバーへの属性を有効にすることができます。 |

## Singularity 環境

[Singularity](https://singularity.lbl.gov/index.html) でコンテナーを実行している場合は、上記の変数の前に **SINGULARITYENV_** を付けることで環境変数を渡すことができます。Singularity 環境変数の詳細については、[こちら](https://singularity.lbl.gov/docs-environment-metadata#environment) を参照してください。

## AWS での実行

AWS でバッチジョブを実行している場合は、W&B 認証情報でマシンを簡単に認証できます。[設定ページ](https://app.wandb.ai/settings) から API key を取得し、[AWS バッチジョブ仕様](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters) で `WANDB_API_KEY` 環境変数を設定します。
