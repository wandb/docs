---
title: Deploy W&B Platform On-premises
description: W&B サーバー をオンプレミス の インフラストラクチャー 上でホストする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-bare-metal
    parent: self-managed
weight: 5
---

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドのデプロイメントオプションを推奨します。W&B のフルマネージドサービスは、シンプルで安全に使用でき、必要な設定は最小限、または不要です。
{{% /alert %}}

関連する質問については、W&B のセールスチーム ([contact@wandb.com](mailto:contact@wandb.com)) にお問い合わせください。

## インフラストラクチャーのガイドライン

W&B のデプロイメントを開始する前に、[リファレンスアーキテクチャー]({{< relref path="ref-arch.md#infrastructure-requirements" lang="ja" >}})、特にインフラストラクチャーの要件を参照してください。

## MySQL データベース

{{% alert color="secondary" %}}
W&B では、MySQL 5.7 の使用は推奨されません。MySQL 5.7 を使用している場合は、最新バージョンの W&B Server との最適な互換性を得るために、MySQL 8 に移行してください。現在、W&B Server は `MySQL 8` のバージョン `8.0.28` 以降のみをサポートしています。
{{% /alert %}}

スケーラブルな MySQL データベースの運用を簡単にするエンタープライズサービスが多数あります。W&B では、以下のソリューションのいずれかを検討することをお勧めします。

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

W&B Server MySQL 8.0 を実行する場合、または MySQL 5.7 から 8.0 にアップグレードする場合は、以下の条件を満たしてください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0 での `sort_buffer_size` の処理方法の変更により、`sort_buffer_size` パラメータをデフォルト値の `262144` から更新する必要がある場合があります。W&B と MySQL が効率的に連携するように、値を `67108864` (64MiB) に設定することをお勧めします。MySQL は v8.0.28 以降でこの設定をサポートしています。

### データベースに関する考慮事項

次の SQL クエリで、データベースと ユーザーを作成します。`SOME_PASSWORD` は任意のパスワードに置き換えてください。

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

{{% alert %}}
これは、SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしていません。
{{% /alert %}}

### パラメータグループの設定

データベースのパフォーマンスを調整するために、次のパラメータグループが設定されていることを確認してください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## オブジェクトストレージ
オブジェクトストレージは、[Minio cluster](https://docs.min.io/minio/k8s/) で外部ホストすることも、署名付き URL をサポートする Amazon S3 互換のオブジェクトストレージでホストすることもできます。オブジェクトストレージが署名付き URL をサポートしているかどうかを確認するには、[次のスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) を実行してください。

また、オブジェクトストレージには、次の CORS ポリシーを適用する必要があります。

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

Amazon S3 互換のオブジェクトストレージに接続するときに、接続文字列で認証情報を指定できます。たとえば、次のように指定できます。

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

オブジェクトストレージに信頼できる SSL 証明書を設定している場合は、W&B が TLS 経由でのみ接続するように指示することもできます。これを行うには、URL に `tls` クエリパラメータを追加します。たとえば、次の URL の例は、Amazon S3 URI に TLS クエリパラメータを追加する方法を示しています。

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

{{% alert color="secondary" %}}
これは、SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしていません。
{{% /alert %}}

サードパーティのオブジェクトストレージを使用する場合は、`BUCKET_QUEUE` を `internal://` に設定します。これにより、W&B サーバーは、外部の SQS キューまたは同等のものに依存する代わりに、すべてのオブジェクト通知を内部で管理するように指示されます。

独自のオブジェクトストレージを実行する際に考慮すべき最も重要な点は次のとおりです。

1. **ストレージ容量とパフォーマンス**。磁気ディスクを使用しても問題ありませんが、これらのディスクの容量を監視する必要があります。W&B の平均的な使用量では、10 ギガバイトから 100 ギガバイトになります。ヘビーな使用では、ペタバイト単位のストレージを消費する可能性があります。
2. **フォールトトレランス**。少なくとも、オブジェクトを保存する物理ディスクは RAID アレイ上にある必要があります。minio を使用する場合は、[分散モード](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed) で実行することを検討してください。
3. **可用性**。ストレージが利用可能であることを確認するために、監視を設定する必要があります。

独自のオブジェクトストレージサービスを実行する代わりに、次のようなエンタープライズの代替手段が多数あります。

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO のセットアップ

minio を使用する場合は、次のコマンドを実行して バケット を作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## W&B Server アプリケーションを Kubernetes にデプロイする

推奨されるインストール方法は、公式の W&B Helm チャートを使用する方法です。W&B Server アプリケーションをデプロイするには、[このセクション]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#deploy-wb-with-helm-cli" lang="ja" >}}) に従ってください。

### OpenShift

W&B は、[OpenShift Kubernetes cluster](https://www.redhat.com/en/technologies/cloud-computing/openshift) 内からの運用をサポートしています。

{{% alert %}}
W&B では、公式の W&B Helm チャートを使用してインストールすることをお勧めします。
{{% /alert %}}

#### 特権のない ユーザー として コンテナ を実行する

デフォルトでは、コンテナ は `$UID` 999 を使用します。オーケストレーターが非 root ユーザー で コンテナ を実行する必要がある場合は、`$UID` >= 100000 および `$GID` 0 を指定します。

{{% alert %}}
W&B は、ファイルシステムの権限が適切に機能するように、root グループ (`$GID=0`) として起動する必要があります。
{{% /alert %}}

Kubernetes のセキュリティコンテキストの例を次に示します。

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## ネットワーク

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを停止する ロードバランサー を実行します。

一般的な ロードバランサー には次のものがあります。
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

機械学習 ペイロード を実行するために使用されるすべてのマシンと、Web ブラウザー を介してサービスにアクセスするために使用されるデバイスが、この エンドポイント と通信できることを確認してください。

### SSL / TLS

W&B Server は SSL を停止しません。セキュリティポリシーで信頼できるネットワーク内での SSL 通信が必要な場合は、Istio や [side car containers](https://istio.io/latest/docs/reference/config/networking/sidecar/) などの ツール の使用を検討してください。ロードバランサー 自体が有効な証明書で SSL を終端する必要があります。自己署名証明書の使用はサポートされておらず、ユーザー に多くの問題を引き起こします。可能であれば、[Let's Encrypt](https://letsencrypt.org) のようなサービスを使用すると、ロードバランサー に信頼できる証明書を提供できます。Caddy や Cloudflare などのサービスは、SSL を管理します。

### Nginx の設定例

以下は、Nginx をリバースプロキシとして使用した設定例です。

```nginx
events {}
http {
    # If we receive X-Forwarded-Proto, pass it through; otherwise, pass along the
    # scheme used to connect to this server
    # X-Forwarded-Proto を受信した場合は、そのまま渡します。それ以外の場合は、このサーバーへの接続に使用されたスキームを渡します。
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # Also, in the above case, force HTTPS
    # また、上記の場合、HTTPS を強制します
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # If we receive X-Forwarded-Host, pass it though; otherwise, pass along $http_host
    # X-Forwarded-Host を受信した場合は、そのまま渡します。それ以外の場合は、$http_host を渡します。
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # If we receive X-Forwarded-Port, pass it through; otherwise, pass along the
    # server port the client connected to
    # X-Forwarded-Port を受信した場合は、そのまま渡します。それ以外の場合は、クライアントが接続したサーバーポートを渡します。
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # If we receive Upgrade, set Connection to "upgrade"; otherwise, delete any
    # Connection header that may have been passed to this server
    # Upgrade を受信した場合は、Connection を "upgrade" に設定します。それ以外の場合は、このサーバーに渡された可能性のある Connection ヘッダーを削除します。
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

## インストール の検証

W&B Server が正しく設定されていることを確認します。ターミナル で次の コマンド を実行します。

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

ログ ファイル を調べて、W&B Server の起動時に発生した エラー を確認します。次の コマンド を実行します。

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

エラー が発生した場合は、W&B Support にお問い合わせください。
