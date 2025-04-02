---
title: Configure environment variables
description: W&B サーバー のインストールを設定する方法
menu:
  default:
    identifier: ja-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

System Settings 管理 UI を使用してインスタンスレベルの 設定 を行うことに加えて、W&B は 環境変数 を使用してコードでこれらの 値 を 設定 する 方法 も提供します。また、[IAM の 高度な 設定 ]({{< relref path="./iam/advanced_env_vars.md" lang="ja" >}})も参照してください。

## 環境変数 のリファレンス

| 環境変数                         | 説明                                                                                                                                                                             |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | wandb/ローカルライセンス                                                                                                                                                           |
| MYSQL                            | MySQL 接続文字列                                                                                                                                                                     |
| BUCKET                           | データ を 保存 する S3 / GCS バケット                                                                                                                                                    |
| BUCKET_QUEUE                     | オブジェクト 作成 イベント 用の SQS / Google PubSub キュー                                                                                                                              |
| NOTIFICATIONS_QUEUE              | run イベント を パブリッシュする SQS キュー                                                                                                                                             |
| AWS_REGION                       | バケット が存在する AWS リージョン                                                                                                                                                     |
| HOST                             | インスタンス の FQD。例えば、`https://my.domain.net` です。                                                                                                       |
| OIDC_ISSUER                      | Open ID Connect ID プロバイダーの URL。例えば、`https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd` です。 |
| OIDC_CLIENT_ID                   | ID プロバイダー内の アプリケーション の クライアント ID                                                                                                                                   |
| OIDC_AUTH_METHOD                 | implicit (デフォルト) または pkce。詳細については、以下を参照してください。                                                                                                                                   |
| SLACK_CLIENT_ID                  | アラートに使用する Slack アプリケーション の クライアント ID                                                                                                                        |
| SLACK_SECRET                     | アラートに使用する Slack アプリケーション の シークレット                                                                                                                           |
| LOCAL_RESTORE                    | インスタンス に アクセス できない場合は、一時的に true に 設定 できます。一時的な 認証情報 については、コンテナ からの ログ を確認してください。                                              |
| REDIS                            | W&B で 外部 REDIS インスタンス を セットアップするために使用できます。                                                                                                                              |
| LOGGING_ENABLED                  | true に 設定 すると、 アクセス ログ が stdout に ストリーム されます。この 変数 を 設定 しなくても、サイドカー コンテナ をマウントして `/var/log/gorilla.log` を監視できます。                              |
| GORILLA_ALLOW_USER_TEAM_CREATION | true に 設定 すると、管理者以外の ユーザー が 新しい Teams を 作成 できるようになります。デフォルトでは false です。                                                                                                         |
| GORILLA_DATA_RETENTION_PERIOD | 削除された run の データ を保持する期間 (時間単位)。削除された run データ は 復元できません。入力 値 に `h` を追加します。たとえば、`"24h"`。 |
| ENABLE_REGISTRY_UI               |  true に 設定 すると、新しい W&B Registry UI が有効になります。            |

{{% alert %}}
GORILLA_DATA_RETENTION_PERIOD 環境変数 は 慎重に使用してください。環境変数 が 設定 されると、データ は 直ちに削除されます。また、このフラグを有効にする前に、データベース と ストレージ バケット の両方を バックアップ することをお勧めします。
{{% /alert %}}

## 高度な 信頼性 設定

### Redis

外部 Redis サーバー の 設定 はオプションですが、 production システム では推奨されます。Redis は、サービスの信頼性を向上させ、特に 大規模な Projects での ロード時間 を短縮するために キャッシュ を有効にするのに役立ちます。高可用性 (HA) を備えた ElastiCache などの マネージド Redis サービス と、次の 仕様 を使用してください。

- 最小 4GB の メモリ、推奨 8GB
- Redis バージョン 6.x
- 転送中の 暗号化
- 認証 が有効

W&B で Redis インスタンス を 設定 するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定 ページ に移動します。[外部 Redis インスタンス を 使用 する] オプション を有効にし、次の 形式 で Redis 接続文字列 を入力します。

{{< img src="/images/hosting/configure_redis.png" alt="W&B で REDIS を 設定 する" >}}

コンテナ または Kubernetes デプロイメント で 環境変数 `REDIS` を使用して Redis を 設定 することもできます。または、`REDIS` を Kubernetes シークレット として 設定 することもできます。

この ページ では、Redis インスタンス が デフォルト ポート `6379` で 実行されていることを前提としています。別の ポート を 設定 し、 認証 を 設定 し、`redis` インスタンス で TLS を有効にする 場合 、接続文字列 の 形式 は `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true` のようになります。
