---
description: W&Bサーバーのインストール方法の設定
displayed_sidebar: default
---


# 環境変数

W&Bは、システム設定の管理UIを介してインスタンスレベルの設定を構成することに加え、環境変数を使用してこれらの値をコードで設定する方法も提供しています。また、[IAMの高度な設定](./iam/advanced_env_vars.md)も参照してください。

## コードとしての設定

| 環境変数                         | 説明                                                                                                                                                                                     |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | あなたの wandb/local ライセンス                                                                                                                                                          |
| MYSQL                            | MySQL 接続文字列                                                                                                                                                                         |
| BUCKET                           | データを保存するためのS3 / GCSバケット                                                                                                                                                     |
| BUCKET_QUEUE                     | オブジェクト作成イベントのためのSQS / Google PubSubキュー                                                                                                                                 |
| NOTIFICATIONS_QUEUE              | run イベントを公開するためのSQSキュー                                                                                                                                                     |
| AWS_REGION                       | あなたのバケットが存在するAWSリージョン                                                                                                                                                   |
| HOST                             | インスタンスのFQD、例：[https://my.domain.net](https://my.domain.net)                                                                                                                    |
| OIDC_ISSUER                      | Open ID ConnectアイデンティティプロバイダーのURL、例：[https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd](https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd) |
| OIDC_CLIENT_ID                   | アイデンティティプロバイダー内のアプリケーションのクライアントID                                                                                                                          |
| OIDC_AUTH_METHOD                 | Implicit（デフォルト）またはpkce、詳細は以下を参照してください                                                                                                                              |
| SLACK_CLIENT_ID                  | アラート用に使用するSlackアプリケーションのクライアントID                                                                                                                                  |
| SLACK_SECRET                     | アラート用に使用するSlackアプリケーションのシークレット                                                                                                                                    |
| LOCAL_RESTORE                    | インスタンスにアクセスできない場合、一時的にこれをtrueに設定することができます。コンテナのログから一時的な資格情報を確認してください。                                               |
| REDIS                            | W&Bで外部のRedisインスタンスを設定するために使用できます。                                                                                                                               |
| LOGGING_ENABLED                  | trueに設定すると、アクセスログがstdoutにストリーミングされます。また、サイドカーコンテナをマウントし、この変数を設定せずに`/var/log/gorilla.log`をテールすることもできます。                   |
| GORILLA_ALLOW_USER_TEAM_CREATION | trueに設定すると、非管理者ユーザーが新しいチームを作成することを許可します。デフォルトはfalseです。                                                                                       |
| GORILLA_DATA_RETENTION_PERIOD | run から削除されたデータを保持する期間（時間単位）。削除された run データは回復できません。入力値に `h` を追加します。例えば、`"24h"` です。                                                 |



:::info

GORILLA_DATA_RETENTION_PERIOD環境変数は慎重に使用してください。環境変数が設定されるとすぐにデータが削除されます。このフラグを有効にする前に、データベースとストレージバケットの両方をバックアップすることをお勧めします。

:::

## 高度な信頼性設定

#### Redis

外部のRedisサーバーを設定することは任意ですが、プロダクション環境システムには強く推奨されます。Redisはサービスの信頼性を向上させ、キャッシュを有効にして読み込み時間を短縮します。特に大規模なProjectsでは効果的です。以下のスペックを持つ高可用性（HA）のマネージドRedisサービス（例：ElastiCache）を使用することをお勧めします：

- 最低4GBのメモリ、推奨8GB
- Redisバージョン6.x
- 転送中の暗号化
- 認証が有効

#### W&BサーバーでのREDISの設定

RedisインスタンスをW&Bに設定するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` のW&B設定ページに移動します。"Use an external Redis instance"オプションを有効にして、`redis`接続文字列を以下の形式で入力します：

![Configuring REDIS in W&B](/images/hosting/configure_redis.png)

コンテナ上またはKubernetesデプロイメントで環境変数`REDIS`を使用して`redis`を設定することもできます。あるいは、`REDIS`をKubernetesシークレットとして設定することもできます。

上記は`redis`インスタンスがデフォルトポート`6379`で実行されていることを前提としています。異なるポートを設定し、認証をセットアップし、さらに`redis`インスタンスでTLSを有効にしたい場合、接続文字列の形式は次のようになります：`redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`