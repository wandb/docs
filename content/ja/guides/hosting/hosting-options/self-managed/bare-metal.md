---
title: Deploy W&B Platform On-premises
description: オンプレミス インフラストラクチャーでの W&B サーバー のホスティング
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-bare-metal
    parent: self-managed
weight: 5
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) デプロイメントタイプなどのフルマネージドデプロイメントオプションをお勧めします。W&B のフルマネージドサービスは、設定が最小限または不要で、シンプルかつ安全に使用できます。
{{% /alert %}}

関連する質問については、W&B 営業チームにお問い合わせください: [contact@wandb.com](mailto:contact@wandb.com)。

## インフラストラクチャーガイドライン

W&B をデプロイする前に、[リファレンスアーキテクチャー]({{< relref path="ref-arch.md#infrastructure-requirements" lang="ja" >}}) 、特にインフラストラクチャー要件を確認してください。

## MySQL データベース

{{% alert color="secondary" %}}
W&B は MySQL 5.7 を使用することをお勧めしません。MySQL 5.7 を使用している場合は、最新バージョンの W&B サーバーとの最高の互換性を確保するために MySQL 8 に移行してください。W&B サーバーは現在、`MySQL 8` バージョン `8.0.28` 以上のみをサポートしています。
{{% /alert %}}

スケーラブルな MySQL データベースの運用を簡素化するエンタープライズサービスがいくつかあります。W&B は次のいずれかのソリューションを検討することをお勧めします:

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

W&B サーバー MySQL 8.0 を実行する場合、または MySQL 5.7 から 8.0 にアップグレードする場合は、以下の条件を満たしてください:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0 が `sort_buffer_size` を処理する方法のいくつかの変更のために、デフォルトの値 `262144` から `sort_buffer_size` パラメータを更新する必要があるかもしれません。MySQL が W&B と効率的に機能することを確実にするために、値を `67108864` (64MiB) に設定することをお勧めします。MySQL はこの設定を v8.0.28 からサポートしています。

### データベースの考慮事項

次の SQL クエリを使用してデータベースとユーザーを作成します。`SOME_PASSWORD` を選択したパスワードに置き換えてください:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

{{% alert %}}
これは SSL 証明書が信頼されている場合にのみ動作します。W&B は自己署名証明書をサポートしていません。
{{% /alert %}}

### パラメーターグループ設定

データベースのパフォーマンスをチューニングするために、次のパラメーターグループが設定されていることを確認してください:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## オブジェクトストア

オブジェクトストアは、外部ホストされた [Minio クラスター](https://docs.min.io/minio/k8s/) または署名付き URL をサポートする Amazon S3 互換のオブジェクトストアに存在することができます。あなたのオブジェクトストアが署名付き URL をサポートしているかを確認するために、[次のスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) を実行します。

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

Amazon S3 互換のオブジェクトストアに接続する際に、接続文字列で資格情報を指定できます。例えば、次のように指定できます:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

オブジェクトストアのために信頼された SSL 証明書を設定する場合、オプションで W&B に TLS を介してのみ接続するよう指示できます。これを行うには、`tls` クエリパラメーターを URL に追加します。例えば、次の URL 例は、Amazon S3 URI に TLS クエリパラメーターを追加する方法を示しています:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

{{% alert color="secondary" %}}
これは、SSL 証明書が信頼されている場合にのみ動作します。W&B は自己署名証明書をサポートしていません。
{{% /alert %}}

サードパーティのオブジェクトストアを使用する場合は、`BUCKET_QUEUE` を `internal://` に設定してください。これにより、外部の SQS キューや同等のものに依存せずに、すべてのオブジェクト通知を W&B サーバーが内部的に管理するようになります。

独自のオブジェクトストアを運用する際に考慮すべき最も重要な事項は次のとおりです:

1. **ストレージ容量とパフォーマンス**: 磁気ディスクを使用しても問題ありませんが、これらのディスクの容量を監視する必要があります。平均的な W&B 使用は 10 GB から 100 GB に達します。重い使用はペタバイトのストレージ消費をもたらす可能性があります。
2. **フォールトトレランス**: 最小限として、オブジェクトを格納する物理ディスクは RAID アレイに配置する必要があります。minio を使用する場合は、[分散モード](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed) で実行することを検討してください。
3. **可用性**: ストレージが利用可能であることを確保するために、モニタリングを構成する必要があります。

独自のオブジェクトストレージサービスを運用する多くのエンタープライズ代替サービスが存在します:

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO の設定

minio を使用する場合、次のコマンドを実行してバケットを作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## W&B サーバーアプリケーションを Kubernetes にデプロイ

推奨されるインストール方法は、公式の W&B Helm チャートを使うことです。[このセクション]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#deploy-wb-with-helm-cli" lang="ja" >}})に従って、W&B サーバーアプリケーションをデプロイしてください。

### OpenShift

W&B は、[OpenShift Kubernetes クラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift) 内からの操作をサポートしています。

{{% alert %}}
W&B 公式の Helm チャートを使用してインストールすることをお勧めします。
{{% /alert %}}

#### 特権のないユーザーとしてコンテナを実行する

デフォルトでは、コンテナは `$UID` 999 を使用します。オーケストレーターが非ルートユーザーでコンテナを実行する必要がある場合は、`$UID` >= 100000 および `$GID` を 0 に指定してください。

{{% alert %}}
W&B は、ファイルシステムパーミッションが正しく機能するために、ルートグループ (`$GID=0`) として開始する必要があります。
{{% /alert %}}

Kubernetes 用のセキュリティコンテキストの例は次のようになります:

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## ネットワーキング

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを停止するロードバランサーを実行します。

一般的なロードバランサーには次のものがあります:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

機械学習のペイロードを実行するために使用されるすべてのマシン、およびウェブブラウザーを通じてサービスにアクセスするために使用されるデバイスがこのエンドポイントと通信できることを確認してください。

### SSL / TLS

W&B サーバーは SSL を停止しません。セキュリティポリシーが信頼されたネットワーク内で SSL 通信を必要とする場合は、Istio や [サイドカーコンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/) のようなツールを使用することを検討してください。ロードバランサー自体は有効な証明書で SSL を終了する必要があります。自己署名証明書の使用はサポートされておらず、ユーザーに多くの課題をもたらす可能性があります。可能であれば、[Let's Encrypt](https://letsencrypt.org) のようなサービスを使用することで、ロードバランサーに信頼できる証明書を提供することができます。Caddy や Cloudflare のようなサービスは SSL を管理してくれます。

### nginx の設定例

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

W&B サーバーが適切に設定されていることを確認してください。次のコマンドをターミナルで実行してください:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

起動時に W&B サーバーがヒットしたエラーを確認するために、ログファイルを確認します。次のコマンドを実行します:

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

エラーが発生した場合は、W&B サポートにお問い合わせください。