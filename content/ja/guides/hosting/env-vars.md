---
title: Configure environment variables
description: W&B サーバー のインストールを設定する方法
menu:
  default:
    identifier: ja-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

W&B では、システム設定の管理者 UI を使用してインスタンスレベルの 設定 を行うだけでなく、 環境変数 を使用して コード でこれらの 値 を 設定 する 方法 も提供しています。また、[IAM の詳細 設定 ]({{< relref path="./iam/advanced_env_vars.md" lang="ja" >}})も参照してください。

## 環境変数 のリファレンス

| 環境変数                         | 説明                                                                                                                                                                                |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | wandb/local のライセンス                                                                                                                                                               |
| MYSQL                            | MySQL の接続文字列                                                                                                                                                              |
| BUCKET                           | データの保存先の S3 / GCS バケット                                                                                                                                                      |
| BUCKET_QUEUE                     | オブジェクト 作成 イベント のための SQS / Google PubSub キュー                                                                                                                              |
| NOTIFICATIONS_QUEUE              | run イベント をパブリッシュする SQS キュー                                                                                                                                               |
| AWS_REGION                       | バケット が存在する AWS リージョン                                                                                                                                                   |
| HOST                             | インスタンス の FQD。例：`https://my.domain.net`                                                                                                                |
| OIDC_ISSUER                      | Open ID Connect アイデンティティプロバイダーへの URL。例：`https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd` |
| OIDC_CLIENT_ID                   | アイデンティティプロバイダーの アプリケーション のクライアント ID                                                                                                                                 |
| OIDC_AUTH_METHOD                 | implicit (デフォルト) または pkce。詳細については下記を参照してください                                                                                                                                   |
| SLACK_CLIENT_ID                  | アラート に使用する Slack アプリケーション のクライアント ID                                                                                                                               |
| SLACK_SECRET                     | アラート に使用する Slack アプリケーション のシークレット                                                                                                                                 |
| LOCAL_RESTORE                    | インスタンス にアクセスできない場合は、一時的に true に 設定 できます。一時的な認証情報については、コンテナ からの ログ を確認してください。                                                                 |
| REDIS                            | W&B で外部 REDIS インスタンス を 設定 するために使用できます。                                                                                                                                |
| LOGGING_ENABLED                  | true に 設定 すると、 アクセス ログ が stdout にストリーミングされます。また、この 変数 を 設定 しなくても、サイドカーコンテナ をマウントして `/var/log/gorilla.log` を監視できます。                                                              |
| GORILLA_ALLOW_USER_TEAM_CREATION | true に 設定 すると、管理者以外の ユーザー も新しい Team を 作成 できます。デフォルトは False です。                                                                                                         |
| GORILLA_DATA_RETENTION_PERIOD | 削除された run の データを保持する時間（時間単位）。削除された run データ は復元できません。入力 値 に `h` を追加します。例： `"24h"`。 |
| ENABLE_REGISTRY_UI               | true に 設定 すると、新しい W&B Registry UI が有効になります。            |

{{% alert %}}
GORILLA_DATA_RETENTION_PERIOD 環境変数 は慎重に使用してください。 環境変数 が 設定 されると、 データ はすぐに削除されます。また、このフラグを有効にする前に、データベースと ストレージ バケット の両方を バックアップ することをお勧めします。
{{% /alert %}}

## 高度な信頼性 設定

### Redis

外部 Redis サーバー の 設定 はオプションですが、 production システム では推奨されます。Redis は、サービスの信頼性を向上させ、特に大規模な Projects での 負荷時間 を短縮するために キャッシュ を有効にするのに役立ちます。高可用性 (HA) を備えた ElastiCache などのマネージド Redis サービス と、次の 仕様 を使用します。

- 最小 4GB のメモリ、推奨 8GB
- Redis バージョン 6.x
- 転送中の暗号化
- 認証が有効

W&B で Redis インスタンス を 設定 するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定 ページに移動します。「外部 Redis インスタンス を使用する」オプションを有効にし、次の 形式 で Redis 接続文字列を入力します。

{{< img src="/images/hosting/configure_redis.png" alt="W&B での REDIS の 設定 " >}}

コンテナ または Kubernetes デプロイメント で 環境変数 `REDIS` を使用して Redis を 設定 することもできます。または、`REDIS` を Kubernetes シークレット として 設定 することもできます。

この ページ では、Redis インスタンス がデフォルトポート `6379` で実行されていることを前提としています。別の ポート を 設定 し、認証を 設定 し、`redis` インスタンス で TLS を有効にする場合は、接続文字列の 形式 は `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true` のようになります。
