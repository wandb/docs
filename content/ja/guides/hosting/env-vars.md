---
title: Configure environment variables
description: W&B サーバー インストールの設定方法
menu:
  default:
    identifier: ja-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

システム設定管理UIを通じてインスタンスレベルの設定を配置することに加えて、W&Bは環境変数を使用してこれらの値をコードで設定する方法も提供しています。また、[IAMの高度な設定]({{< relref path="./iam/advanced_env_vars.md" lang="ja" >}})も参照してください。

## 環境変数リファレンス

| 環境変数                        | 説明                                                                                                                         |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | あなたの wandb/local ライセンス                                                                                              |
| MYSQL                            | MySQL接続文字列                                                                                                              |
| BUCKET                           | データを保存するためのS3 / GCS バケット                                                                                       |
| BUCKET_QUEUE                     | オブジェクト作成イベント用のSQS / Google PubSubキュー                                                                       |
| NOTIFICATIONS_QUEUE              | runイベントを公開するためのSQSキュー                                                                                         |
| AWS_REGION                       | あなたのバケットが存在するAWSリージョン                                                                                       |
| HOST                             | あなたのインスタンスのFQD、例：`https://my.domain.net`                                                                        |
| OIDC_ISSUER                      | あなたのOpen ID Connect IDプロバイダーへのURL、例：`https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd`         |
| OIDC_CLIENT_ID                   | IDプロバイダー内のアプリケーションのクライアントID                                                                           |
| OIDC_AUTH_METHOD                 | 暗黙的（デフォルト）またはpkce、詳細は以下を参照                                                                             |
| SLACK_CLIENT_ID                  | アラート用に使用したいSlackアプリケーションのクライアントID                                                                   |
| SLACK_SECRET                     | アラート用に使用したいSlackアプリケーションのシークレット                                                                     |
| LOCAL_RESTORE                    | インスタンスにアクセスできない場合、一時的にtrueに設定できます。コンテナからのログで一時的な資格情報を確認してください。       |
| REDIS                            | W&Bで外部のREDISインスタンスをセットアップするために使用できます。                                                           |
| LOGGING_ENABLED                  | trueに設定した場合、アクセスログがstdoutにストリーミングされます。また、サイドカーコンテナをマウントし、変数を設定せずに`/var/log/gorilla.log`を監視することもできます。 |
| GORILLA_ALLOW_USER_TEAM_CREATION | trueに設定した場合、非管理者ユーザーが新しいチームを作成できるようになります。デフォルトはfalseです。                        |
| GORILLA_DATA_RETENTION_PERIOD    | 削除したrunからのデータを保持する期間（時間）。削除されたrunデータは復元できません。入力値に `h` を追加してください。例えば、`"24h"`。 |
| ENABLE_REGISTRY_UI               | trueに設定した場合、新しいW&B Registry UIを有効にします。                                                                       |

{{% alert %}}
GORILLA_DATA_RETENTION_PERIOD 環境変数は慎重に使用してください。環境変数が設定されるとすぐにデータが削除されます。このフラグを有効にする前に、データベースとストレージバケットの両方をバックアップすることをお勧めします。
{{% /alert %}}

## 高度な信頼性設定

### Redis

外部のRedisサーバーの設定は任意ですが、プロダクションシステムには推奨されます。Redisはサービスの信頼性を向上させ、特に大規模プロジェクトにおいてロード時間を短縮するためにキャッシングを可能にします。HA（高可用性）と以下の仕様を備えたElastiCacheのようなマネージドRedisサービスを使用します：

- 最小4GBのメモリ、推奨8GB
- Redis バージョン 6.x
- イン・トランジット暗号化
- 認証有効化

RedisインスタンスをW&Bで設定するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin`でW&B設定ページに移動します。「外部Redisインスタンスを使用する」オプションを有効にし、次の形式でRedis接続文字列を入力します：

{{< img src="/images/hosting/configure_redis.png" alt="W&BでのREDIS設定" >}}

コンテナ上またはKubernetesデプロイメントで環境変数`REDIS`を使用してRedisを設定することもできます。あるいはKubernetesのシークレットとして`REDIS`を設定することもできます。

このページはRedisインスタンスがデフォルトのポート`6379`で動作していることを前提としています。異なるポートを設定し、認証を設定し、`redis`インスタンスでTLSを有効にしたい場合、接続文字列の形式は次のようになります： `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`