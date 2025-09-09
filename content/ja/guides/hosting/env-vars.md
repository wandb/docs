---
title: 環境変数の設定
description: W&B サーバー インストールの設定方法
menu:
  default:
    identifier: ja-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

W&B では、System Settings 管理 UI からインスタンスレベルの設定を行えるだけでなく、環境変数を使ってコードからこれらの値を設定することもできます。あわせて、[IAM の高度な設定]({{< relref path="./iam/advanced_env_vars.md" lang="ja" >}}) も参照してください。

## 環境変数リファレンス

| 環境変数             | 説明                                                                                                                                                                              |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LICENSE`                          | wandb/local のライセンス                                                                                                                                                                 |
| `MYSQL`                            | MySQL 接続文字列                                                                                                                                                              |
| `BUCKET`                           | データを保存するための S3 / GCS バケット                                                                                                                                                     |
| `BUCKET_QUEUE`                     | オブジェクト作成イベントのための SQS / Google PubSub キュー                                                                                                                                 |
| `NOTIFICATIONS_QUEUE`              | run イベントを公開するための SQS キュー                                                                                                                                             |
| `AWS_REGION`                       | バケットが存在する AWS リージョン                                                                                                                                                   |
| `HOST`                             | インスタンスの FQD。例: `https://my.domain.net`                                                                                                       |
| `OIDC_ISSUER`                      | Open ID Connect の ID プロバイダーへの URL。例: `https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd` |
| `OIDC_CLIENT_ID`                   | ID プロバイダー内のアプリケーションのクライアント ID                                                                                                                                   |
| `OIDC_AUTH_METHOD`                 | Implicit (デフォルト) または pkce。詳細は以下を参照してください                                                                                                                                   |
| `SLACK_CLIENT_ID`                  | アラートに使用する Slack アプリケーションのクライアント ID                                                                                                                        |
| `SLACK_SECRET`                     | アラートに使用する Slack アプリケーションのシークレット                                                                                                                           |
| `LOCAL_RESTORE`                    | インスタンスにアクセスできない場合は、一時的に `true` に設定できます。暫定的な認証情報はコンテナーのログを参照してください。                                              |
| `REDIS`                            | W&B で外部 Redis インスタンスを設定するために使用します。                                                                                                                                |
| `LOGGING_ENABLED`                  | `true` に設定すると、アクセスログが stdout にストリームされます。この変数を設定しなくても、サイドカーコンテナーをマウントして `/var/log/gorilla.log` をテールすることもできます。                              |
| `GORILLA_ALLOW_USER_TEAM_CREATION` | `true` に設定すると、管理者以外の Users が新しい Teams を作成できるようになります。デフォルトは `false` です。                                                                                                         |
| `GORILLA_CUSTOMER_SECRET_STORE_SOURCE` | W&B Weave で使用されるチームシークレットを保存するためのシークレットマネージャーを設定します。以下のシークレットマネージャーがサポートされています: <ul><li><b>内部シークレットマネージャー</b> (デフォルト): <code>k8s-secretmanager://wandb-secret</code></li><li><b>AWS Secret Manager</b>: <code>aws-secretmanager</code></li><li><b>GCP Secret Manager</b>: <code>gcp-secretmanager</code></li><li><b>Azure</b>: <code>az-secretmanger</code></li></ul>  |
| `GORILLA_DATA_RETENTION_PERIOD`    | 削除された run データを保持する期間を時間単位で指定します。削除された run データは復元できません。入力値に `h` を付けます。例: `"24h"`。 |
| `GORILLA_DISABLE_PERSONAL_ENTITY`  | `true` に設定すると、[パーソナル Entities]({{< relref path="/support/kb-articles/difference_team_entity_user_entity_mean_me.md" lang="ja" >}}) が無効になります。新しいパーソナル Projects の作成や、既存のパーソナル Projects への書き込みが防止されます。 |
| `ENABLE_REGISTRY_UI`               | `true` に設定すると、新しい W&B Registry UI を有効にします。            |
| `WANDB_ARTIFACT_DIR`               | ダウンロードしたすべての Artifacts の保存先。未設定の場合は、トレーニングスクリプトの `artifacts` ディレクトリーがデフォルトになります。このディレクトリーが存在し、実行中のユーザーに書き込み権限があることを確認してください。これは生成されたメタデータファイルの保存場所は制御しません。メタデータの保存場所は `WANDB_DIR` 環境変数で設定できます。 |
| `WANDB_DATA_DIR`                   | ステージングした Artifacts をアップロードする場所。デフォルトの場所はプラットフォームによって異なります。これは `platformdirs` Python パッケージの `user_data_dir` の値を使用するためです。このディレクトリーが存在し、実行中のユーザーに書き込み権限があることを確認してください。 |
| `WANDB_DIR`                        | 生成されたすべてのファイルの保存先。未設定の場合は、トレーニングスクリプトの `wandb` ディレクトリーがデフォルトになります。このディレクトリーが存在し、実行中のユーザーに書き込み権限があることを確認してください。これはダウンロードした Artifacts の保存場所は制御しません。Artifacts の保存場所は `WANDB_ARTIFACT_DIR` 環境変数で設定できます。 |
| `WANDB_IDENTITY_TOKEN_FILE`        | [ID フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}}) の場合、Java Web Tokens (JWT) が保存されるローカルディレクトリーへの絶対パス。 |
{{% alert %}}
GORILLA_DATA_RETENTION_PERIOD 環境変数は慎重に使用してください。この環境変数を設定すると、データは直ちに削除されます。フラグを有効にする前に、データベースとストレージバケットの両方をバックアップすることを推奨します。
{{% /alert %}}

## 高度な信頼性設定

### Redis

外部 Redis サーバーの設定はオプションですが、プロダクションシステムでは推奨されます。Redis はサービスの信頼性を高め、特に大規模な Projects でロード時間を短縮するためのキャッシュを有効にするのに役立ちます。高可用性 (HA) を備えた ElastiCache のようなマネージド Redis サービスの利用と、以下の仕様を推奨します:
- 最小 4GB のメモリ、推奨 8GB
- Redis バージョン 6.x
- 転送中の暗号化
- 認証が有効

W&B で Redis インスタンスを構成するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定ページに移動します。「外部 Redis インスタンスを使用」オプションを有効にし、以下の形式で Redis 接続文字列を入力します。

{{< img src="/images/hosting/configure_redis.png" alt="W&B での Redis の設定" >}}

コンテナーまたは Kubernetes のデプロイメントで、環境変数 `REDIS` を使用して Redis を構成することもできます。あるいは、`REDIS` を Kubernetes シークレットとして設定することもできます。
このページは、Redis インスタンスがデフォルトポート `6379` で実行されていることを前提としています。異なるポートを構成し、認証を設定し、`redis` インスタンスで TLS を有効にしたい場合、接続文字列の形式は `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true` のようになります。