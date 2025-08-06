---
title: 環境変数を設定する
description: W&B サーバーのインストールを設定する方法
menu:
  default:
    identifier: ja-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

システム設定管理 UI からインスタンスレベルの設定を行うだけでなく、W&B では環境変数を使ってコードからこれらの値を設定する方法も提供しています。さらに、[IAM の高度な設定]({{< relref path="./iam/advanced_env_vars.md" lang="ja" >}})も参照してください。

## 環境変数リファレンス

| 環境変数                          | 説明                                                                                                                                                 |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LICENSE`                         | ご利用中の wandb/local ライセンス                                                                                                                    |
| `MYSQL`                           | MySQL の接続文字列                                                                                                                                   |
| `BUCKET`                          | データ保存用の S3 / GCS バケット                                                                                                                     |
| `BUCKET_QUEUE`                    | オブジェクト作成イベント用の SQS / Google PubSub キュー                                                                                              |
| `NOTIFICATIONS_QUEUE`             | run イベントを公開する SQS キュー                                                                                                                    |
| `AWS_REGION`                      | バケットが存在する AWS リージョン                                                                                                                    |
| `HOST`                            | インスタンスの FQDN、例: `https://my.domain.net`                                                                                 |
| `OIDC_ISSUER`                     | Open ID Connect の ID プロバイダーの URL、例: `https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd`           |
| `OIDC_CLIENT_ID`                  | ID プロバイダー内のアプリケーション Client ID                                                                                                        |
| `OIDC_AUTH_METHOD`                | Implicit（デフォルト）または pkce。詳細は下記を参照してください                                                                                       |
| `SLACK_CLIENT_ID`                 | アラートに使う Slack アプリケーションの クライアント ID                                                                                             |
| `SLACK_SECRET`                    | アラートに使う Slack アプリケーションの シークレット                                                                                                 |
| `LOCAL_RESTORE`                   | インスタンスにアクセスできない場合、一時的に true に設定可能。コンテナのログから一時的な認証情報を確認してください。                                  |
| `REDIS`                           | W&B で外部 REDIS インスタンスを設定する場合に使用                                                                                                    |
| `LOGGING_ENABLED`                 | true に設定するとアクセスログが stdout にストリーミングされます。サイドカーコンテナをマウントし、この変数を設定せずに `/var/log/gorilla.log` を tail することも可能です。 |
| `GORILLA_ALLOW_USER_TEAM_CREATION` | true で管理者以外のユーザーによる新規 team 作成を許可します。デフォルトは false です。                                                             |
| `GORILLA_CUSTOMER_SECRET_STORE_SOURCE` | W&B Weave で使用される team シークレットを保存するシークレットマネージャーを指定します。以下のシークレットマネージャーがサポートされています： <ul><li><b>Internal secret manager</b>（デフォルト）：<code>k8s-secretmanager://wandb-secret</code></li><li><b>AWS Secret Manager</b>：<code>aws-secretmanager</code></li><li><b>GCP Secret Manager</b>：<code>gcp-secretmanager</code></li><li><b>Azure</b>：<code>az-secretmanger</code></li></ul> |
| `GORILLA_DATA_RETENTION_PERIOD`   | run から削除されたデータを残しておく期間（時間単位）。削除された run データは復元できません。入力値の末尾に `h` を付けてください（例： `"24h"`）。        |
| `GORILLA_DISABLE_PERSONAL_ENTITY` | true に設定すると[個人 entities]({{< relref path="/support/kb-articles/difference_team_entity_user_entity_mean_me.md" lang="ja" >}})が無効化されます。個人 entities 内での新しい個人 Projects の作成と、既存個人 Projects への書き込みを防止します。|
| `ENABLE_REGISTRY_UI`              | true に設定すると新しい W&B Registry UI が有効になります。                                                                                         |
| `WANDB_ARTIFACT_DIR`              | ダウンロードした artifacts の保存先。未設定の場合、トレーニングスクリプトからの相対パスで `artifacts` ディレクトリがデフォルトになります。ディレクトリが存在し、実行ユーザーに書き込み権限があることを確認してください。この変数は生成されるメタデータファイルの保存場所は制御しません。これらは `WANDB_DIR` 環境変数で指定可能です。  |
| `WANDB_DATA_DIR`                  | ステージングアーティファクトのアップロード先。デフォルトの場所はプラットフォームによって異なり、Python パッケージ `platformdirs` の `user_data_dir` の値が使用されます。ディレクトリが存在し、実行ユーザーに書き込み権限があることを確認してください。    |
| `WANDB_DIR`                       | 生成されたすべてのファイルの保存先。未設定の場合、トレーニングスクリプトからの相対パスで `wandb` ディレクトリがデフォルトになります。ディレクトリが存在し、実行ユーザーに書き込み権限があることを確認してください。この変数はダウンロードした artifacts の保存場所は制御しません。これらは `WANDB_ARTIFACT_DIR` 環境変数で指定可能です。  |
| `WANDB_IDENTITY_TOKEN_FILE`       | [ID フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}})用。Java Web Tokens (JWT) が保存されるローカルディレクトリの絶対パス。                |

{{% alert %}}
`GORILLA_DATA_RETENTION_PERIOD` 環境変数は慎重に使用してください。この変数を設定するとデータは即座に削除されます。フラグを有効にする前に、必ずデータベースとストレージバケットの両方のバックアップを取得することを推奨します。
{{% /alert %}}

## 高度な信頼性設定

### Redis

外部 Redis サーバーの設定は任意ですが、プロダクション環境では推奨されます。Redis は、サービスの信頼性向上やキャッシュによる処理速度の短縮（特に大規模 Projects で有効）に貢献します。ElastiCache など、以下の仕様の高可用性（HA）を持つマネージド Redis サービスの利用をおすすめします。

- メモリは最低 4GB、推奨 8GB
- Redis バージョン 6.x
- 転送時暗号化対応
- 認証有効化

W&B で Redis インスタンスを設定するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定ページにアクセスしてください。「外部 Redis インスタンスを使用する」オプションを有効化し、下記の形式で Redis 接続文字列を入力します。

{{< img src="/images/hosting/configure_redis.png" alt="W&B での REDIS 設定" >}}

環境変数 `REDIS` をコンテナや Kubernetes デプロイメントで設定することで Redis を構成することもできます。また、`REDIS` を Kubernetes のシークレットとして設定することも可能です。

このページでは、Redis インスタンスがデフォルトのポート `6379` で稼働していることを想定しています。別のポートを使用する場合、認証を設定し、かつ `redis` インスタンスで TLS を有効にしたい場合の接続文字列の例は次の通りです：
`redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`