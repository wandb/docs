---
title: 環境変数を設定する
description: W&B サーバーのインストールを設定する方法
menu:
  default:
    identifier: env-vars
    parent: w-b-platform
weight: 7
---

システム設定管理 UI からインスタンスレベルの設定を行う以外にも、W&B では環境変数を使ってコードからこれらの値を設定することもできます。詳しくは [IAM の高度な設定]({{< relref "./iam/advanced_env_vars.md" >}}) もご覧ください。

## 環境変数リファレンス

| 環境変数                            | 説明                                                                                                                                                                              |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LICENSE`                           | あなたの wandb/local ライセンス                                                                                                                                                     |
| `MYSQL`                             | MySQL の接続文字列                                                                                                                                                                 |
| `BUCKET`                            | データ保存用の S3 / GCS バケット                                                                                                                                                   |
| `BUCKET_QUEUE`                      | オブジェクト作成イベント用の SQS / Google PubSub キュー                                                                                                                             |
| `NOTIFICATIONS_QUEUE`               | run イベント用に発行する SQS キュー                                                                                                                                                |
| `AWS_REGION`                        | バケットが存在する AWS リージョン                                                                                                                                                   |
| `HOST`                              | インスタンスの FQDN。例：`https://my.domain.net`                                                                                              |
| `OIDC_ISSUER`                       | Open ID Connect アイデンティティプロバイダーの URL。例：`https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd`                     |
| `OIDC_CLIENT_ID`                    | アイデンティティプロバイダー上のアプリケーションのクライアント ID                                                                                                                     |
| `OIDC_AUTH_METHOD`                  | Implicit（デフォルト）または pkce。詳細は下記参照                                                                                                                                   |
| `SLACK_CLIENT_ID`                   | アラートに使用したい Slack アプリケーションのクライアント ID                                                                                                                         |
| `SLACK_SECRET`                      | アラートに使用したい Slack アプリケーションのシークレット                                                                                                                            |
| `LOCAL_RESTORE`                     | インスタンスにアクセスできない場合、一時的に true に設定可能。コンテナのログから一時認証情報を確認してください。                                                                  |
| `REDIS`                             | W&B で外部の REDIS インスタンスを設定する場合に使用できます。                                                                                                                        |
| `LOGGING_ENABLED`                   | true に設定するとアクセスログが stdout にストリームされます。変数を設定しなくても、サイドカーコンテナをマウントして `/var/log/gorilla.log` を tail することが可能です。         |
| `GORILLA_ALLOW_USER_TEAM_CREATION`  | true に設定すると、非管理者ユーザーが新しい team を作成できるようになります（デフォルトは false）。                                                                              |
| `GORILLA_CUSTOMER_SECRET_STORE_SOURCE` | W&B Weave が使用する team シークレットの保存先シークレットマネージャー。サポートされているシークレットマネージャー：<ul><li><b>Internal secret manager</b>（デフォルト）：<code>k8s-secretmanager://wandb-secret</code></li><li><b>AWS Secret Manager</b>：<code>aws-secretmanager</code></li><li><b>GCP Secret Manager</b>：<code>gcp-secretmanager</code></li><li><b>Azure</b>：<code>az-secretmanger</code></li><ul>  |
| `GORILLA_DATA_RETENTION_PERIOD`     | run から削除されたデータを保持する期間（時間単位）。削除された run データは復元できません。入力値の末尾に `h` を付与してください（例："24h"）。            |
| `GORILLA_DISABLE_PERSONAL_ENTITY`   | true に設定すると、[personal entities]({{< relref "/support/kb-articles/difference_team_entity_user_entity_mean_me.md" >}}) を無効化します。個人 entities 内での新規プロジェクト作成や既存プロジェクトへの書き込みができなくなります。 |
| `ENABLE_REGISTRY_UI`                | true に設定すると、新しい W&B Registry UI を有効化します。            |
| `WANDB_ARTIFACT_DIR`                | ダウンロードした全 Artifacts を保存するディレクトリー。未設定時はトレーニングスクリプト直下の `artifacts` ディレクトリーとなります。このディレクトリーが存在し、かつ実行ユーザーに書き込み権限があることを確認してください。この変数は生成されるメタデータファイルの保存場所は制御しません。メタデータファイルは `WANDB_DIR` 環境変数で保存先を設定できます。      |
| `WANDB_DATA_DIR`                    | ステージング Artifacts のアップロード先。プラットフォームによってデフォルト保存先が異なります（`platformdirs` Python パッケージの `user_data_dir` 値を使用）。このディレクトリーが存在し、かつ実行ユーザーに書き込み権限があることを確認してください。 |
| `WANDB_DIR`                         | 全ての生成ファイルの保存先。未設定時はトレーニングスクリプト直下の `wandb` ディレクトリーとなります。このディレクトリーが存在し、かつ実行ユーザーに書き込み権限があることを確認してください。この変数は Artifacts のダウンロード先は制御しません。ダウンロード先は `WANDB_ARTIFACT_DIR` 環境変数で設定できます。    |
| `WANDB_IDENTITY_TOKEN_FILE`         | [identity federation]({{< relref "/guides/hosting/iam/authentication/identity_federation.md" >}}) 用の、Java Web Token（JWT）を保存するローカルディレクトリへの絶対パス。        |
{{% alert %}}
`GORILLA_DATA_RETENTION_PERIOD` 環境変数は慎重にご利用ください。この環境変数を設定すると、データは即座に削除されます。フラグを有効にする前に、データベースとストレージバケットの両方をバックアップすることを強く推奨します。
{{% /alert %}}

## 高度な信頼性設定

### Redis

外部 Redis サーバーの設定は任意ですが、プロダクション環境では推奨されます。Redis を導入することでサービスの信頼性が向上し、特に大規模プロジェクトでのキャッシュによる応答速度の向上に役立ちます。以下の条件を満たし、高可用性（HA）対応の ElastiCache などのマネージド Redis サービスの利用がおすすめです。

- メモリ最低 4GB（推奨 8GB）
- Redis バージョン 6.x
- 転送時の暗号化
- 認証有効化

W&B で Redis インスタンスを設定するには、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定ページにアクセスし、"Use an external Redis instance" オプションを有効化し、次の形式で Redis 接続文字列を入力します：

{{< img src="/images/hosting/configure_redis.png" alt="W&B での REDIS 設定" >}}

また、コンテナや Kubernetes デプロイメントで環境変数 `REDIS` を使用して Redis を設定することも可能です。あるいは、`REDIS` を Kubernetes シークレットとしてセットアップする方法もあります。

本ページでは Redis インスタンスがデフォルトポート `6379` で稼働していることを前提としています。別のポートを使用する場合や認証・TLS を有効にしたい場合は、接続文字列は `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true` のようになります。