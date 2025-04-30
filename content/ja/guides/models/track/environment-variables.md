---
title: 環境変数
description: W&B 環境変数を設定します。
menu:
  default:
    identifier: ja-guides-models-track-environment-variables
    parent: experiments
weight: 9
---

スクリプトを自動化された環境で実行するとき、スクリプトの実行前またはスクリプト内で設定された環境変数を使って **wandb** を制御できます。

```bash
# これは秘密であり、バージョン管理にチェックインするべきではありません
WANDB_API_KEY=$YOUR_API_KEY
# 名前とメモはオプションです
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# wandb/settingsファイルをチェックインしない場合にのみ必要です
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# スクリプトをクラウドに同期したくなければ
os.environ["WANDB_MODE"] = "offline"

# Sweep IDの追跡をRunオブジェクトと関連クラスに追加
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## オプションの環境変数

これらのオプションの環境変数を使用して、リモートマシンでの認証をセットアップすることができます。

| 変数名 | 使用法 |
| --------------------------- | ---------- |
| **WANDB_ANONYMOUS** | `allow`, `never`, または `must` に設定して、ユーザーが秘密のURLで匿名のrunsを作成できるようにします。 |
| **WANDB_API_KEY** | あなたのアカウントに関連付けられた認証キーを設定します。キーは[設定ページ](https://app.wandb.ai/settings)で確認できます。このキーは、リモートマシンで `wandb login` が実行されていない場合に設定する必要があります。 |
| **WANDB_BASE_URL** | [wandb/local]({{< relref path="/guides/hosting/" lang="ja" >}})を使用している場合、この環境変数を `http://YOUR_IP:YOUR_PORT` に設定してください。 |
| **WANDB_CACHE_DIR** | これはデフォルトで \~/.cache/wandb に設定されています。この環境変数を使用してこの場所を上書きすることができます。 |
| **WANDB_CONFIG_DIR** | これはデフォルトで \~/.config/wandb に設定されています。この環境変数を使用してこの場所を上書きすることができます。 |
| **WANDB_CONFIG_PATHS** | カンマで区切られたyamlファイルのリストをwandb.configにロードします。[config]({{< relref path="./config.md#file-based-configs" lang="ja" >}})を参照してください。 |
| **WANDB_CONSOLE** | stdout / stderr ロギングを無効にする場合はこれを "off" に設定します。これが機能する環境では、デフォルトで "on" に設定されています。 |
| **WANDB_DATA_DIR** | ステージングアーティファクトがアップロードされる場所。デフォルトの場所はプラットフォームに依存し、`platformdirs` Pythonパッケージからの `user_data_dir` の値を使用します。 |
| **WANDB_DIR** | すべての生成されたファイルを _wandb_ ディレクトリーではなくここに保存するために絶対パスを設定します。_このディレクトリーが存在しており、プロセスが実行されるユーザーが書き込めることを確認してください_。この設定は、ダウンロードされたアーティファクトの場所には影響しません、それらの場所を設定するには代わりに _WANDB_ARTIFACT_DIR_ を使用してください。 |
| **WANDB_ARTIFACT_DIR** | すべてのダウンロードされたアーティファクトを _artifacts_ ディレクトリーではなくここに保存するために絶対パスを設定します。このディレクトリーが存在しており、プロセスが実行されるユーザーが書き込めることを確認してください。この設定は、生成されたメタデータファイルの場所には影響しません、それらの場所を設定するには代わりに _WANDB_DIR_ を使用してください。 |
| **WANDB_DISABLE_GIT** | gitリポジトリをプローブし、最新のコミット / 差分をキャプチャするのを防ぎます。 |
| **WANDB_DISABLE_CODE** | ノートブックやgit差分の保存を防ぐためにtrueに設定します。gitリポジトリ内にいる場合、依然として現在のコミットを保存します。 |
| **WANDB_DOCKER** | dockerイメージのダイジェストを設定してrunsの復元を有効にします。これはwandb dockerコマンドで自動的に設定されます。イメージダイジェストを取得するには `wandb docker my/image/name:tag --digest` を実行します。 |
| **WANDB_ENTITY** | あなたのrunに関連付けられたentityです。トレーニングスクリプトのディレクトリーで `wandb init` を実行した場合、_wandb_ という名前のディレクトリーが作成され、デフォルトのentityが保存され、ソース管理にチェックインされます。このファイルを作成したくない場合やファイルを上書きしたい場合、環境変数を使用できます。 |
| **WANDB_ERROR_REPORTING** | wandbが致命的なエラーをエラートラッキングシステムにログするのを防ぎたい場合はfalseに設定します。 |
| **WANDB_HOST** | システムが提供するホスト名を使用せずにwandbインターフェースで表示したいホスト名を設定します。 |
| **WANDB_IGNORE_GLOBS** | 無視するファイルのグロブのカンマ区切りリストを設定します。これらのファイルはクラウドに同期されません。 |
| **WANDB_JOB_NAME** | `wandb` が作成するジョブに対する名前を指定します。 |
| **WANDB_JOB_TYPE** | ジョブタイプを指定します。「トレーニング」や「評価」など、異なるタイプのrunsを示します。詳細については[grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}})を参照してください。 |
| **WANDB_MODE** | "offline" に設定すると、wandbはrunメタデータをローカルに保存し、サーバーに同期しなくなります。`disabled` に設定すると、wandbは完全にオフになります。 |
| **WANDB_NAME** | あなたのrunの人間が読める名前。設定されていない場合、ランダムに生成されます。 |
| **WANDB_NOTEBOOK_NAME** | jupyterで実行されている場合、この変数でノートブックの名前を設定できます。これを自動検出しようとします。 |
| **WANDB_NOTES** | あなたのrunに関する長いメモ。Markdownが許可されており、UIで後で編集できます。 |
| **WANDB_PROJECT** | あなたのrunに関連付けられたプロジェクトです。これも `wandb init` で設定できますが、環境変数は値を上書きします。 |
| **WANDB_RESUME** | デフォルトでは _never_ に設定されています。_auto_ に設定すると、wandbは自動的に失敗したrunsを再開します。_must_ に設定すると、開始時に必ずrunが存在するように強制します。常に独自のユニークなIDを生成したい場合、_allow_ に設定して常に **WANDB_RUN_ID** を設定してください。 |
| **WANDB_RUN_GROUP** | 自動的にrunsをまとめるためのexperiment nameを指定します。詳細については[grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}})を参照してください。 |
| **WANDB_RUN_ID** | スクリプトの単一runに対応する、グローバルにユニークな文字列（プロジェクトごとに）を設定します。64文字以内でなければなりません。すべての非単語文字はダッシュに変換されます。失敗時の既存のrunを再開するために使用できます。 |
| **WANDB_SILENT** | wandbログステートメントを黙らせるために **true** に設定します。これを設定すると、すべてのログが **WANDB_DIR**/debug.log に書き込まれます。 |
| **WANDB_SHOW_RUN** | あなたのオペレーティングシステムがサポートしていれば、runのURLを自動的にブラウザで開くために **true** に設定します。 |
| **WANDB_SWEEP_ID** | `Run` オブジェクトおよび関連クラスにSweep IDの追跡を追加し、UIに表示します。 |
| **WANDB_TAGS** | runに適用されるタグのカンマ区切りリスト。 |
| **WANDB_USERNAME** | runに関連付けられたあなたのチームメンバーのユーザー名。これは、サービスアカウントAPIキーと共に使用して、自動化されたrunsをあなたのチームのメンバーに帰属させることができます。 |
| **WANDB_USER_EMAIL** | runに関連付けられたあなたのチームメンバーのメール。これはサービスアカウントAPIキーと共に使用して、自動化されたrunsをあなたのチームのメンバーに帰属させることができます。 |

## Singularity 環境

[Singularity](https://singularity.lbl.gov/index.html)でコンテナを実行している場合、上記の変数に **SINGULARITYENV_** をプレフィックスとしてつけて環境変数を渡すことができます。Singularity環境変数に関する詳細は[こちら](https://singularity.lbl.gov/docs-environment-metadata#environment)で確認できます。

## AWSでの実行

AWSでバッチジョブを実行している場合、W&Bのクレデンシャルでマシンを認証するのは簡単です。[設定ページ](https://app.wandb.ai/settings)からAPIキーを取得し、[AWSバッチジョブ仕様](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters)で `WANDB_API_KEY` 環境変数を設定します。