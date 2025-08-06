---
title: wandb ローンンチ
---

**使用方法**

`wandb launch [OPTIONS]`

**概要**

W&B ジョブをローンチまたはキューに登録します。詳細はこちら：https://wandb.me/launch

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-u, --uri (str)` | ローカルパスまたは git リポジトリの uri を指定してローンチします。指定した場合、このコマンドはその uri からジョブを作成します。 |
| `-j, --job (str)` | ローンチするジョブの名前。指定した場合、ローンチ時に uri は不要です。 |
| `--entry-point` | Project 内でのエントリーポイント。[デフォルト: main]。エントリーポイントが見つからない場合、指定された名前の Project ファイルをスクリプトとして実行します（.py ファイルは python で、.sh ファイルは 環境 変数 $SHELL で指定したデフォルトシェルで実行）。指定された場合、設定ファイルで渡された entrypoint の値を上書きします。 |
| `--build-context (str)` | ソース コード 内でのビルドコンテキストまでのパス。デフォルトはソース コード のルート。-u と組み合わせてのみ使用可能です。 |
| `--name` | ローンチする run の名前。未指定の場合はランダムな run 名でローンチされます。指定した場合、設定ファイルで渡された name の値を上書きします。 |
| `-e, --entity (str)` | 新しい run を送信する Entities 名。デフォルトはローカルの wandb/settings フォルダで設定されている Entities を使用します。指定した場合、設定ファイルで渡された entity の値を上書きします。 |
| `-p, --project (str)` | 新しい run を送信する Projects 名。デフォルトは source uri から与えられた Project 名、または github run の場合は git リポジトリ名。指定した場合、設定ファイルで渡された project の値を上書きします。 |
| `-r, --resource` | run の実行リソース。利用可能な値: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex'。リソース設定のないキューへ push する場合は必須パラメータです。指定した場合、設定ファイルの resource の値を上書きします。 |
| `-d, --docker-image` | 使用したい特定の docker イメージ（name:tag 形式）。指定した場合、設定ファイルで指定された docker image の値を上書きします。 |
| `--base-image` | ジョブの コード を実行する Docker イメージ。--docker-image とは併用不可。 |
| `-c, --config` | .json で終わる JSON ファイルのパスまたは JSON 文字列を指定し、ローンチの設定として渡します。ローンチされる run の設定を決定します。 |
| `-v, --set-var` | 許可リスト有効なキュー向けテンプレート変数の値（キーと値のペア）。例: `--set-var key1=value1 --set-var key2=value2` |
| `-q, --queue` | run を push する run キューの名前。未指定の場合は単一 run を直接ローンチします。引数を指定せず `--queue` のみを使うと 'default' キューになります。名前を指定した場合、その run キューは渡した project・entity 配下に存在している必要があります。 |
| `--async` | ジョブを非同期で実行するフラグ。デフォルトは false（--async を使わない限り、wandb launch はジョブ終了まで待機します）。このオプションは --queue とは併用できません。エージェント で実行する際の非同期オプションは wandb launch-agent 側で設定してください。 |
| `--resource-args` | .json で終わる JSON ファイルのパスまたは JSON 文字列。計算リソース用の resource args として渡されます。具体的な内容は各実行バックエンドによって異なります。詳細はドキュメントをご参照ください。 |
| `--dockerfile` | ジョブビルド時に使用する Dockerfile のパス（ジョブのルートからの相対パス） |
| `--priority [critical|high|medium|low]` | --queue を指定した場合、そのジョブの優先度を設定します。優先度が高いジョブから順に実行されます。順序は critical → high → medium → low です。 |