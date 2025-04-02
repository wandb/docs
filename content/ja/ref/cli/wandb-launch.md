---
title: wandb launch
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch
---

**使用方法**

`wandb launch [OPTIONS]`

**概要**

W&B Job を Launch またはキューに入れます。https://wandb.me/launch を参照してください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-u, --uri (str)` | Launch するローカルパスまたは git リポジトリ URI。これが指定されると、指定された URI からジョブが作成されます。 |
| `-j, --job (str)` | Launch するジョブの名前。これが渡されると、Launch は URI を必要としません。 |
| `--entry-point` | プロジェクト内のエントリポイント。[default: main]。エントリポイントが見つからない場合、指定された名前のプロジェクトファイルをスクリプトとして実行しようとします。.py ファイルを実行するには「python」を使用し、.sh ファイルを実行するにはデフォルトシェル（環境変数 $SHELL で指定）を使用します。これが渡されると、設定ファイルを使用して渡されたエントリポイントの値を上書きします。 |
| `--build-context (str)` | ソースコード内のビルドコンテキストへのパス。デフォルトはソースコードのルートです。-u とのみ互換性があります。 |
| `--name` | run を Launch する run の名前。指定しない場合、ランダムな run 名が run の Launch に使用されます。これが渡されると、設定ファイルを使用して渡された名前を上書きします。 |
| `-e, --entity (str)` | 新しい run の送信先となるターゲット Entity の名前。デフォルトでは、ローカルの wandb/settings フォルダで設定された Entity が使用されます。これが渡されると、設定ファイルを使用して渡された Entity の値を上書きします。 |
| `-p, --project (str)` | 新しい run の送信先となるターゲット Project の名前。デフォルトでは、ソース URI によって指定された Project 名、または github run の場合は git リポジトリ名が使用されます。これが渡されると、設定ファイルを使用して渡された Project の値を上書きします。 |
| `-r, --resource` | run に使用する実行リソース。サポートされている値：「local-process」、「local-container」、「kubernetes」、「sagemaker」、「gcp-vertex」。これは、リソース構成なしでキューにプッシュする場合に必須のパラメータになりました。これが渡されると、設定ファイルを使用して渡されたリソースの値を上書きします。 |
| `-d, --docker-image` | 使用したい特定の Docker イメージ。name:tag の形式。これが渡されると、設定ファイルを使用して渡された Docker イメージの値を上書きします。 |
| `--base-image` | ジョブコードを実行する Docker イメージ。--docker-image と互換性がありません。 |
| `-c, --config` | JSON ファイル（「.json」で終わる必要があります）または Launch 設定として渡される JSON 文字列へのパス。Launch された run の構成方法を指示します。 |
| `-v, --set-var` | 許可リストが有効になっているキューのテンプレート変数の値を、キーと値のペアとして設定します。例：`--set-var key1=value1 --set-var key2=value2` |
| `-q, --queue` | プッシュ先の run キューの名前。ない場合は、単一の run を直接 Launch します。引数なしで指定された場合（`--queue`）、デフォルトではキュー「default」になります。それ以外の場合、名前が指定されている場合は、指定された run キューが、指定された Project および Entity の下に存在する必要があります。 |
| `--async` | ジョブを非同期で実行するためのフラグ。デフォルトは false です。つまり、--async が設定されていない限り、wandb launch はジョブが完了するまで待機します。このオプションは --queue と互換性がありません。エージェント での実行時に非同期オプションは wandb launch-agent で設定する必要があります。 |
| `--resource-args` | コンピュートリソースにリソース引数として渡される JSON ファイル（「.json」で終わる必要があります）または JSON 文字列へのパス。提供する必要がある正確な内容は、実行バックエンドごとに異なります。このファイルのレイアウトについては、ドキュメントを参照してください。 |
| `--dockerfile` | ジョブの構築に使用される Dockerfile へのパス（ジョブルートからの相対パス）。 |
| `--priority [critical|high|medium|low]` | --queue が渡された場合、ジョブの優先度を設定します。優先度の高い Launch ジョブが最初に処理されます。優先度の高い順に、critical、high、medium、low となります。 |
