
# wandb launch

**使用方法**

`wandb launch [オプション]`

**概要**

W&B Jobを起動またはキューに追加します。詳細は https://wandb.me/launch を参照してください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| -u, --uri (str) | 起動するためのローカルパスまたはgitリポジトリのuri。指定された場合、このコマンドは指定されたuriからジョブを作成します。 |
| -j, --job (str) | 起動するジョブの名前。指定された場合、uriは必要ありません。 |
| --entry-point | プロジェクト内のエントリーポイント。[デフォルト: main]。エントリーポイントが見つからない場合、指定された名前のプロジェクトファイルをスクリプトとして実行しようとします。.pyファイルは 'python' で、.shファイルはデフォルトのシェル（環境変数 $SHELL で指定）で実行します。指定された場合、configファイルで渡されたentrypointの値を上書きします。 |
| --build-context (str) | ソースコード内のビルドコンテキストへのパス。デフォルトはソースコードのルート。-u と互換性あり。 |
| --name | runを起動する際のrunの名前。指定されていない場合、ランダムなrun名が使用されます。指定された場合、configファイルで渡された名前を上書きします。 |
| -e, --entity (str) | 新しいrunが送られる対象のEntityの名前。デフォルトはローカルのwandb/settingsフォルダで設定されたentityを使用します。指定された場合、configファイルで渡されたentityの値を上書きします。 |
| -p, --project (str) | 新しいrunが送られる対象のProjectの名前。デフォルトでは、ソースuriによって与えられたプロジェクト名や、githubのrunの場合はgitリポジトリ名を使用します。指定された場合、configファイルで渡されたProject値を上書きします。 |
| -r, --resource | runを実行するためのリソース。サポートされている値: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex'。これは、リソース設定がない状態でキューに投入する場合に必須のパラメータです。指定された場合、configファイルで渡されたresourceの値を上書きします。 |
| -d, --docker-image | 使用したい特定のdockerイメージ。形式は name:tag。指定された場合、configファイルで渡されたdockerイメージの値を上書きします。 |
| -c, --config | JSONファイルへのパス（'.json'で終わる必要があります）またはJSON文字列。起動configとして渡され、起動されるrunの設定を決定します。 |
| -v, --set-var | 許可リストが有効になっているキューのテンプレート変数値をキーと値のペアで設定します。例：`--set-var key1=value1 --set-var key2=value2` |
| -q, --queue | キューに投入するrunの名前。指定がない場合は、単一のrunを直接起動します。引数なしで指定（`--queue`）された場合、デフォルトのキューとして 'default' を使用します。それ以外の場合、指定された名前のrunキューは指定されたProjectとEntityの下に存在する必要があります。 |
| --async | ジョブを非同期で実行するためのフラグ。デフォルトはfalse、つまり--asyncが設定されていない限り、wandb launchはジョブの終了を待ちます。このオプションは--queueと互換性がありません。非同期オプションはエージェント実行時に wandb launch-agent で設定する必要があります。 |
| --resource-args | JSONファイルへのパス（'.json'で終わる必要があります）またはJSON文字列。計算リソースに渡されるリソース引数として渡されます。提供すべき具体的な内容は各実行バックエンドによって異なります。このファイルのレイアウトについてはドキュメントを参照してください。 |
| --dockerfile | ジョブをビルドするために使用されるDockerfileのパス。ジョブのルートからの相対パス。 |
| --priority [critical|high|medium|low] | --queueが指定された場合、ジョブの優先順位を設定します。優先順位が高いほど先に処理されます。優先順位の順序は、高から低の順に: critical, high, medium, low。 |