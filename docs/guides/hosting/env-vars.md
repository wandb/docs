---
description: How to configure the W&B Server installation
displayed_sidebar: default
---

# 環境変数

システム設定の管理UIを通じてインスタンスレベルの設定を行うことに加えて、W&Bは環境変数を使用してコードでこれらの値を設定する方法も提供しています。

## コードによる設定

| 環境変数              | 説明                                                                                                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LICENSE              | wandb/localライセンス                                                                                                                                                                 |
| MYSQL                | MySQL接続文字列                                                                                                                                                                       |
| BUCKET               | データを保存するためのS3 / GCS バケット                                                                                                                                                     |
| BUCKET_QUEUE         | オブジェクト作成イベントのためのSQS / Google PubSubキュー                                                                                                                                |
| NOTIFICATIONS_QUEUE  | runイベントを公開するためのSQSキュー                                                                                                                                                      |
| AWS_REGION           | バケットが存在するAWSリージョン                                                                                                                                                      |
| HOST                 | インスタンスのFQD、例：[https://my.domain.net](https://my.domain.net)                                                                                                            |
| OIDC_ISSUER          | Open ID ConnectアイデンティティプロバイダへのURL、例: [https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd](https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd) |
| OIDC_CLIENT_ID       | アイデンティティプロバイダ内のアプリケーションのクライアントID                                                                                                                                   |
| OIDC_AUTH_METHOD     | 暗黙的（デフォルト）またはpkce、詳細は以下のコンテキストを参照                                                                                                                                   |
| SLACK_CLIENT_ID      | アラートに使用するSlackアプリケーションのクライアントID                                                                                                                                    |
| SLACK_SECRET         | アラートに使用するSlackアプリケーションのシークレット                                                                                                                                    |
| LOCAL_RESTORE        | インスタンスにアクセスできない場合、一時的にtrueに設定できます。コンテナからのログを確認して一時的な資格情報を取得してください。                                              |
| REDIS                | 外部REDISインスタンスをW&Bで設定するために使用できます。                                                                                                                               |
| LOGGING_ENABLED      | trueに設定されると、アクセスログがstdoutにストリーミングされます。この変数を設定せずにサイドカーコンテナをマウントし、`/var/log/gorilla.log` を追跡することもできます。        |

## 高度な信頼性設定
#### Redis

外部のRedisサーバーを設定することはオプションですが、プロダクションシステムでは強くお勧めします。Redisを使用すると、サービスの信頼性が向上し、特に大規模なプロジェクトでは読み込み時間が短縮されるキャッシュが利用できるようになります。高可用性（HA）を持つマネージドRedisサービス（例：ElastiCache）を使用することをお勧めします。以下の仕様が必要です。

- 最低4GBのメモリ、推奨8GB

- Redisバージョン6.x

- 通信中の暗号化

- 認証が有効

#### W&BサーバーでREDISを設定する

RedisインスタンスをW&Bと組み合わせて設定するには、W&Bの設定ページにアクセスしてください。そのURLは、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` です。「外部のRedisインスタンスを使用する」オプションを有効にし、以下の形式で`redis`接続文字列を入力します。

![W&BでREDISを設定する](/images/hosting/configure_redis.png)

また、コンテナやKubernetesデプロイメントで環境変数`REDIS`を使って`redis`を設定することもできます。あるいは、`REDIS`をKubernetesのシークレットとして設定することもできます。

上記は、`redis`インスタンスがデフォルトのポート`6379`で実行されていることを前提としています。異なるポートを設定し、認証をセットアップし、`redis`インスタンスでTLSを有効にする場合、接続文字列の形式は次のようになります：`redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`