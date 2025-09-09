---
title: wandb launch
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch
---

**使い方**

`wandb launch [OPTIONS]`

**概要**

W&B Job を Launch するか、キューに入れます。https://wandb.me/launch を参照してください

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-u, --uri (str)` | Launch するためのローカルパスまたは Git リポジトリの URI。指定した場合、このコマンドはその URI から Job を作成します。 |
| `-j, --job (str)` | Launch する Job の名前。指定した場合、Launch に URI は不要です。 |
| `--entry-point` | project 内のエントリーポイント。[デフォルト: main]。エントリーポイントが見つからない場合、指定した名前の project ファイルをスクリプトとして実行しようとします。.py ファイルは 'python'、.sh ファイルはデフォルトのシェル（環境変数 $SHELL で指定）で実行します。指定した場合、設定ファイルで渡された entrypoint の値を上書きします。 |
| `--build-context (str)` | ソースコード内のビルドコンテキストへのパス。デフォルトはソースコードのルート。-u のみで使用可能です。 |
| `--name` | run を Launch する際に付与する run の名前。指定しない場合はランダムな run 名が使われます。指定した場合、設定ファイルで渡された name を上書きします。 |
| `-e, --entity (str)` | 新しい run の送信先となる対象 entity の名前。デフォルトではローカルの wandb/settings フォルダで設定された entity を使用します。指定した場合、設定ファイルで渡された entity の値を上書きします。 |
| `-p, --project (str)` | 新しい run の送信先となる対象 project の名前。デフォルトでは source URI から得られる project 名、または GitHub の run の場合は Git リポジトリ名を使用します。指定した場合、設定ファイルで渡された project の値を上書きします。 |
| `-r, --resource` | run に使用する実行リソース。サポートされる値: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex'。リソースの設定がない queue にプッシュする場合、このパラメータは必須です。指定した場合、設定ファイルで渡された resource の値を上書きします。 |
| `-d, --docker-image` | 使用したい特定の Docker イメージ。形式は name:tag。指定した場合、設定ファイルで渡された Docker イメージの値を上書きします。 |
| `--base-image` | Job の コード を実行する Docker イメージ。--docker-image とは併用不可です。 |
| `-c, --config` | Launch の設定として渡される JSON ファイル（拡張子は '.json' である必要があります）のパス、または JSON 文字列。起動された run の設定方法を指定します。 |
| `-v, --set-var` | 許可リスト（allow listing）が有効な queue に対して、テンプレート変数の値をキーと値のペアとして設定します。例: `--set-var key1=value1 --set-var key2=value2` |
| `-q, --queue` | プッシュ先の run キューの名前。指定しない場合は単一の run を直接 Launch します。引数なし（`--queue`）で指定した場合は、キュー 'default' がデフォルトになります。名前を指定した場合は、その run キューが指定した project と entity の配下に存在する必要があります。 |
| `--async` | Job を非同期に実行するフラグ。デフォルトは false、つまり --async を設定しない限り、wandb launch は Job の完了を待機します。このオプションは --queue と併用できません。エージェント で実行する際の非同期設定は wandb launch-agent 側で指定してください。 |
| `--resource-args` | コンピュートリソースへの resource 引数として渡される JSON ファイル（拡張子は '.json' である必要があります）のパス、または JSON 文字列。提供すべき正確な内容は実行バックエンドごとに異なります。このファイルのレイアウトはドキュメントを参照してください。 |
| `--dockerfile` | Job をビルドするのに使用する Dockerfile へのパス（Job のルートからの相対パス）。 |
| `--priority [critical|high|medium|low]` | --queue 指定時に Job の優先度を設定します。優先度が高い Launch の Job から先に処理されます。優先度の高い順は: critical, high, medium, low です。 |