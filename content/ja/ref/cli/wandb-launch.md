---
title: wandb ローンチ
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch
---

**使い方**

`wandb launch [OPTIONS]`

**概要**

W&B Job をローンチまたはキューに追加します。詳しくは https://wandb.me/launch をご覧ください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-u, --uri (str)` | ローカルパスまたは git リポジトリの URI を指定してローンチします。指定した場合、このコマンドはその URI から Job を作成します。 |
| `-j, --job (str)` | ローンチする Job の名前。指定した場合、URI は不要です。 |
| `--entry-point` | プロジェクト内のエントリーポイント。[デフォルト: main] エントリーポイントが見つからない場合、指定した名前のプロジェクトファイルをスクリプトとして実行しようとします。.py ファイルは 'python' で、.sh ファイルは環境変数 $SHELL で指定されたデフォルトシェルを使って実行します。指定した場合、設定ファイル内で指定された entrypoint の値を上書きします。 |
| `--build-context (str)` | ソースコード内のビルドコンテキストのパス。デフォルトはソースコードのルートです。-u と併用可能です。 |
| `--name` | この run の名前。指定しない場合はランダムな run 名が使われます。指定した場合、設定ファイル内の name を上書きします。 |
| `-e, --entity (str)` | 新しい run を送信する Entities の名前。デフォルトではローカルの wandb/settings フォルダで設定された entity が使われます。指定した場合、設定ファイル内の entity の値を上書きします。 |
| `-p, --project (str)` | 新しい run を送信する Projects の名前。デフォルトは source uri で指定された project 名、または github run の場合は git リポジトリ名です。指定した場合、設定ファイル内の project の値を上書きします。 |
| `-r, --resource` | run に使用する実行リソース。利用可能な値: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex'。リソース設定のないキューにプッシュする場合、必須パラメータです。指定した場合、設定ファイル内の resource の値を上書きします。 |
| `-d, --docker-image` | 使用したい特定の Docker イメージ（name:tag 形式）。指定した場合、設定ファイル内の docker image の値を上書きします。 |
| `--base-image` | Job コードを実行する Docker イメージ。--docker-image とは併用できません。 |
| `-c, --config` | JSON ファイル（'.json' で終わる）または JSON 文字列のパス。ローンチ config として渡され、ローンチされる run の設定内容を指定します。 |
| `-v, --set-var` | allow listing が有効なキューでテンプレート変数の値をキーと値のペアで設定します。例：`--set-var key1=value1 --set-var key2=value2` |
| `-q, --queue` | プッシュする run キューの名前。指定しない場合、run を直接ローンチします。引数なし（`--queue`）の場合は 'default' キューが使われます。名前を指定した場合、その run キューが指定された project と entity に存在している必要があります。 |
| `--async` | ジョブを非同期で実行するフラグ。デフォルトは false なので、--async を指定しない限り wandb launch はジョブの終了まで待機します。このオプションは --queue と互換性がありません。エージェントで実行する際の非同期オプションは wandb launch-agent で設定してください。 |
| `--resource-args` | リソースに渡す JSON ファイル（'.json' で終わる）または JSON 文字列のパス。実行バックエンドごとに内容は異なります。詳細はドキュメントをご確認ください。 |
| `--dockerfile` | ジョブをビルドする際に使用する Dockerfile へのパス（ジョブのルートからの相対パス） |
| `--priority [critical|high|medium|low]` | --queue が指定されている場合、ジョブの優先度を設定します。優先度の高い Launch ジョブが先に実行されます。優先度は critical, high, medium, low の順です。 |