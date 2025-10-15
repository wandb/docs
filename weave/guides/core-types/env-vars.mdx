---
title: 環境変数を設定する
description: W&B サーバー インストールの設定方法
menu:
  default:
    identifier: ja-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

インスタンスレベルの設定を System Settings 管理 UI 経由で設定することに加えて、W&B は環境変数を使用してこれらの値をコードで設定する方法も提供しています。また、[IAM の高度な設定]({{< relref path="./iam/advanced_env_vars.md" lang="ja" >}})も参照してください。

## 環境変数リファレンス

| 環境変数                         | 説明                                                                                                                                                                           |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | あなたの wandb/local ライセンス                                                                                                                                               |
| MYSQL                            | MySQL 接続文字列                                                                                                                                                              |
| BUCKET                           | データを保存するための S3 / GCS バケット                                                                                                                                      |
| BUCKET_QUEUE                     | オブジェクト作成イベントのための SQS / Google PubSub キュー                                                                                                                   |
| NOTIFICATIONS_QUEUE              | run イベントを公開するための SQS キュー                                                                                                                                        |
| AWS_REGION                       | バケットが存在する AWS リージョン                                                                                                                                              |
| HOST                             | インスタンスの FQD、つまり `https://my.domain.net`                                                                                                          |
| OIDC_ISSUER                      | Open ID Connect アイデンティティプロバイダーのURL、つまり `https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd`      |
| OIDC_CLIENT_ID                   | あなたのアイデンティティプロバイダーにあるアプリケーションのクライアントID                                                                                                    |
| OIDC_AUTH_METHOD                 | デフォルトは Implicit で、pkceも可能です。詳細は以下を参照してください。                                                                                                        |
| SLACK_CLIENT_ID                  | アラートに使用したい Slack アプリケーションのクライアント ID                                                                                                                    |
| SLACK_SECRET                     | アラートに使用したい Slack アプリケーションの秘密キー                                                                                                                         |
| LOCAL_RESTORE                    | インスタンスにアクセスできない場合、一時的にこれをtrueに設定できます。コンテナのログを確認して一時的な資格情報を取得してください。                                              |
| REDIS                            | W&B で外部の REDIS インスタンスを設定するために使用できます。                                                                                                                 |
| LOGGING_ENABLED                  | true に設定すると、アクセスログが stdout にストリームされます。また、この変数を設定しなくてもサイドカーコンテナをマウントし、`/var/log/gorilla.log` を追跡できます。          |
| GORILLA_ALLOW_USER_TEAM_CREATION | true に設定すると、非管理者ユーザーが新しいチームを作成できるようになります。デフォルトは false です。                                                                         |
| GORILLA_DATA_RETENTION_PERIOD | 削除された run のデータを何時間保持するか。削除された run データは復元できません。入力値に `h` を付けてください。例えば、`"24h"`。                                        |
| ENABLE_REGISTRY_UI               | true に設定すると、新しい W&B Registry UI が有効になります。                                                                                          |

{{% alert %}}
GORILLA_DATA_RETENTION_PERIOD 環境変数は慎重に使用してください。環境変数が設定されるとすぐにデータは削除されます。また、このフラグを有効にする前に、データベースとストレージバケットの両方をバックアップすることをお勧めします。
{{% /alert %}}

## 高度な信頼性設定

### Redis

外部の Redis サーバーを設定することはオプションですが、プロダクションシステムでは推奨されます。Redis はサービスの信頼性を向上させ、特に大規模プロジェクトでのロード時間を短縮するためのキャッシングを可能にします。ElastiCache などの高可用性 (HA) を備えた管理された Redis サービスを使用し、以下の仕様を備えています:

- 最小 4GB のメモリ、推奨 8GB
- Redis バージョン 6.x
- 転送中の暗号化
- 認証が有効

W&B で Redis インスタンスを設定するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定ページに移動します。「外部 Redis インスタンスを使用する」オプションを有効にし、Redis 接続文字列を次の形式で入力します:

{{< img src="/images/hosting/configure_redis.png" alt="W&B での REDIS の設定" >}}

また、コンテナ上や Kubernetes デプロイメントで環境変数 `REDIS` を使って Redis を設定することもできます。あるいは、`REDIS` を Kubernetes シークレットとして設定することもできます。

このページは、Redis インスタンスがデフォルトのポート `6379` で動作していることを前提としています。異なるポートを設定した場合、認証をセットアップし、さらに Redis インスタンスで TLS を有効にしたい場合、接続文字列の形式は次のようになります: `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`