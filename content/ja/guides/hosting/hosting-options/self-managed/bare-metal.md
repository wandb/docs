---
title: W&B プラットフォームをオンプレミスで展開する
description: オンプレミス インフラストラクチャーでの W&B Server のホスティング
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-bare-metal
    parent: self-managed
weight: 5
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のような完全管理されたデプロイメントオプションを推奨しています。W&B の完全管理サービスは、シンプルで安全に使用でき、ほとんど設定が不要です。
{{% /alert %}}

関連する質問については、W&B セールスチームにお問い合わせください: [contact@wandb.com](mailto:contact@wandb.com).

## インフラストラクチャーガイドライン

W&B のデプロイメントを開始する前に、特にインフラストラクチャー要件を確認するために、[リファレンスアーキテクチャー]({{< relref path="ref-arch.md#infrastructure-requirements" lang="ja" >}})を参照してください。

## MySQL データベース

{{% alert color="secondary" %}}
W&B は MySQL 5.7 の使用を推奨しません。MySQL 5.7 を使用している場合は、最新バージョンの W&B Server との互換性を最大限にするために MySQL 8 へ移行してください。W&B Server は現在、`MySQL 8` バージョン `8.0.28` 以降のみをサポートしています。
{{% /alert %}}

スケーラブルな MySQL データベースの運用を簡素化するエンタープライズサービスがいくつかあります。W&B は次のソリューションのいずれかを検討することを推奨します:

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

以下の条件を満たすようにしてください。W&B Server MySQL 8.0 を実行する場合、または MySQL 5.7 から 8.0 へのアップグレード時に以下を行います:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0 で `sort_buffer_size` の扱いにいくつか変更があったため、デフォルトの値である `262144` から `sort_buffer_size` パラメータを更新する必要があるかもしれません。W&B と MySQL が効率よく連携することを保証するために、値を `67108864` (64MiB) に設定することを推奨します。MySQL はこの設定を v8.0.28 からサポートしています。

### データベースに関する考慮事項

次の SQL クエリを使用してデータベースとユーザーを作成します。`SOME_PASSWORD` を希望のパスワードに置き換えてください:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

{{% alert %}}
これは SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしていません。
{{% /alert %}}

### パラメータグループ設定

データベースのパフォーマンスを最適化するために、次のパラメータグループが設定されていることを確認してください:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## オブジェクトストレージ

オブジェクトストアは、[Minio クラスター](https://docs.min.io/minio/k8s/) または署名付き URL をサポートする Amazon S3 互換のオブジェクトストアでホストできます。[次のスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) を実行して、オブジェクトストアが署名付き URL をサポートしているか確認してください。

さらに、次の CORS ポリシーをオブジェクトストアに適用する必要があります。

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>http://YOUR-W&B-SERVER-IP</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedMethod>HEAD</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

Amazon S3 互換のオブジェクトストアに接続する際、自分の資格情報を接続文字列で指定できます。たとえば、次のように指定します:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

オブジェクトストア用の信頼された SSL 証明書を設定している場合、W&B が TLS 経由でのみ接続するように指示することもできます。それには、URL に `tls` クエリパラメータを追加します。たとえば、次の URL 例は、Amazon S3 URI に TLS クエリパラメータを追加する方法を示しています:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

{{% alert color="secondary" %}}
これは SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしていません。
{{% /alert %}}

サードパーティのオブジェクトストアを使用している場合、`BUCKET_QUEUE` を `internal://` に設定します。これにより、外部の SQS キューやそれに相当するものに依存せずに、W&B サーバーがすべてのオブジェクト通知を内部で管理できるようになります。

独自のオブジェクトストアを運用する際に考慮すべき最も重要なことは次のとおりです:

1. **ストレージ容量とパフォーマンス**。磁気ディスクを使用しても構いませんが、これらのディスクの容量を監視している必要があります。平均的な W&B の使用量は 10 ギガバイトから 100 ギガバイトに達します。大量使用はペタバイトのストレージ消費を引き起こす可能性があります。
2. **フォールトトレランス**。最低限、オブジェクトを保存する物理ディスクは RAID アレイにあるべきです。minio を使用する場合は、[分散モード](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed) での実行を検討してください。
3. **可用性**。ストレージが利用可能であることを確認するために監視を設定する必要があります。

独自のオブジェクトストレージサービスを運用するためのエンタープライズ代替策は多数存在します:

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO 設定

minio を使用する場合、次のコマンドを実行してバケットを作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## W&B Server アプリケーションを Kubernetes にデプロイ

公式の W&B Helm チャートを使用することが推奨されるインストール方法です。[こちらのセクション]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#deploy-wb-with-helm-cli" lang="ja" >}})に従って W&B Server アプリケーションをデプロイしてください。

### OpenShift

W&B は、[OpenShift Kubernetes クラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift) 内からの運用をサポートしています。

{{% alert %}}
W&B は公式の W&B Helm チャートでのインストールをお勧めします。
{{% /alert %}}

#### コンテナを非特権ユーザーとして実行

デフォルトでは、コンテナは `$UID` 999 を使用します。オーケストレーターが非ルートユーザーでのコンテナ実行を要求する場合、`$UID` >= 100000 そして `$GID` を 0 に指定します。

{{% alert %}}
W&B はファイルシステム権限が正しく機能するためにルートグループ (`$GID=0`) として開始する必要があります。
{{% /alert %}}

Kubernetes のセキュリティコンテキストの例は次のようになります:

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## ネットワーキング

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを停止するロードバランサーを実行します。

一般的なロードバランサーには以下があります:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

機械学習のペイロードを実行するために使用されるすべてのマシンと、Web ブラウザを介してサービスにアクセスするために使用されるデバイスがこのエンドポイントと通信できることを確認してください。

### SSL / TLS

W&B Server は SSL を停止しません。信頼されたネットワーク内での SSL 通信がセキュリティポリシーで求められている場合、Istio や [サイドカーコンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/) などのツールを使用してください。ロードバランサー自体は有効な証明書で SSL を終了させる必要があります。自己署名証明書はサポートされておらず、ユーザーに多くの問題を引き起こします。可能であれば、[Let's Encrypt](https://letsencrypt.org) のようなサービスを使用してロードバランサーに信頼された証明書を提供することは素晴らしい方法です。Caddy や Cloudflare のようなサービスは SSL を自動的に管理します。

### nginx 設定の例

以下は、nginx をリバースプロキシとして使用する例の設定です。

```nginx
events {}
http {
    # If we receive X-Forwarded-Proto, pass it through; otherwise, pass along the
    # scheme used to connect to this server
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # Also, in the above case, force HTTPS
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # If we receive X-Forwarded-Host, pass it though; otherwise, pass along $http_host
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # If we receive X-Forwarded-Port, pass it through; otherwise, pass along the
    # server port the client connected to
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # If we receive Upgrade, set Connection to "upgrade"; otherwise, delete any
    # Connection header that may have been passed to this server
    map $http_upgrade $proxy_connection {
        default upgrade;
        '' close;
    }

    server {
        listen 443 ssl;
        server_name         www.example.com;
        ssl_certificate     www.example.com.crt;
        ssl_certificate_key www.example.com.key;

        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $proxy_connection;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $proxy_x_forwarded_proto;
        proxy_set_header X-Forwarded-Host $proxy_x_forwarded_host;

        location / {
            proxy_pass  http://$YOUR_UPSTREAM_SERVER_IP:8080/;
        }

        keepalive_timeout 10;
    }
}
```

## インストールの確認

W&B Server が正しく設定されていることを確認してください。次のコマンドをターミナルで実行します:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

開始時に W&B Server にヒットするエラーを表示するためにログファイルを確認してください。次のコマンドを実行します:

{{< tabpane text=true >}}
{{% tab header="Docker" value="docker" %}}
```bash
docker logs wandb-local
```
{{% /tab %}}

{{% tab header="Kubernetes" value="kubernetes"%}}
```bash
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```
{{% /tab %}}
{{< /tabpane >}}

エラーが発生した場合は W&B サポートにお問い合わせください。