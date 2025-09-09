---
title: W&B プラットフォーム を オンプレミス にデプロイ
description: オンプレミスのインフラストラクチャーで W&B サーバーをホスティングする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-bare-metal
    parent: self-managed
weight: 5
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) などのフルマネージドなデプロイメント オプションを推奨します。W&B のフルマネージド サービスはシンプルかつ安全に利用でき、設定は最小限または不要です。
{{% /alert %}}

関連する質問は W&B の営業チームまでご連絡ください: [contact@wandb.com](mailto:contact@wandb.com).

## インフラストラクチャー ガイドライン

W&B のデプロイを開始する前に、特にインフラストラクチャー要件については [リファレンス アーキテクチャー]({{< relref path="ref-arch.md#infrastructure-requirements" lang="ja" >}}) を参照してください。

## MySQL データベース

{{% alert color="secondary" %}}
W&B は MySQL 5.7 の使用を推奨しません。MySQL 5.7 をお使いの場合は、最新の W&B Server との互換性を最大化するため MySQL 8 へ移行してください。W&B Server は現在 `MySQL 8` の `8.0.28` 以降のみをサポートしています。
{{% /alert %}}

W&B Server で MySQL 8.0 を使用する場合、または MySQL 5.7 から 8.0 へアップグレードする場合は、以下の条件を満たしてください:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0 の `sort_buffer_size` の扱いが一部変更されたため、デフォルトの `262144` から `sort_buffer_size` パラメータを更新する必要がある場合があります。W&B との連携で効率よく動作させるため、`67108864` (64MiB) に設定することを推奨します。MySQL は v8.0.28 以降でこの設定をサポートします。

### データベースの考慮事項

次の SQL を実行して、データベースとユーザーを作成します。`SOME_PASSWORD` は任意のパスワードに置き換えてください:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

{{% alert %}}
これは SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしません。
{{% /alert %}}

### パラメータ グループの設定

データベースのパフォーマンスをチューニングするため、以下のパラメータ グループが設定されていることを確認してください:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## オブジェクト ストレージ
オブジェクト ストアは、[Minio クラスター](https://min.io/docs/minio/kubernetes/upstream/index.html) や、署名付き URL をサポートする任意の Amazon S3 互換オブジェクト ストアで外部ホスティングできます。お使いのオブジェクト ストアが署名付き URL をサポートしているかどうかを確認するには、[次のスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) を実行してください。

加えて、以下の CORS ポリシーをオブジェクト ストアに適用する必要があります。

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

Amazon S3 互換のオブジェクト ストアに接続する際、接続文字列に認証情報を含めることができます。たとえば次のように指定できます:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

オブジェクト ストアに信頼された SSL 証明書を設定している場合、W&B に TLS 経由のみで接続するよう指示することもできます。その場合は、URL に `tls` クエリ パラメータを追加します。以下は Amazon S3 URI に TLS クエリ パラメータを追加する例です:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

{{% alert color="secondary" %}}
これは SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしません。
{{% /alert %}}

サードパーティのオブジェクト ストアを使用する場合は、`BUCKET_QUEUE` を `internal://` に設定してください。これにより、外部の SQS キューや同等の仕組みに依存せず、オブジェクト通知を W&B サーバー が内部的に管理します。

独自のオブジェクト ストアを運用する際の最重要ポイントは次のとおりです:

1. **ストレージ容量とパフォーマンス**。磁気ディスクの使用でも問題ありませんが、ディスク容量は監視してください。平均的な W&B の利用では数十〜数百 GB に達します。ヘビーな利用では PB 級になる可能性もあります。
2. **フォールト トレランス。** 最低限、オブジェクトを保存する物理ディスクは RAID 構成にするべきです。minio を使用する場合は、[分散モード](https://min.io/docs/minio/kubernetes/upstream/operations/concepts/availability-and-resiliency.html#distributed-minio-deployments) を検討してください。
3. **可用性。** ストレージが利用可能な状態であることを確認するため、監視を設定してください。

自前でオブジェクト ストレージ サービスを運用する代替として、エンタープライズ向けの選択肢が複数あります:

1. [Amazon S3 on Outposts](https://aws.amazon.com/s3/outposts/)
2. [NetApp StorageGRID](https://www.netapp.com/data-storage/storagegrid/)

### MinIO のセットアップ

minio を使用する場合、次のコマンドでバケットを作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## W&B Server アプリケーションを Kubernetes にデプロイ

推奨されるインストール方法は、公式の W&B Helm チャートを使用する方法です。[Helm CLI によるデプロイ手順]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#deploy-wb-with-helm-cli" lang="ja" >}}) に従って W&B Server アプリケーションをデプロイしてください。

### OpenShift

W&B は、[OpenShift Kubernetes クラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift) 内からの運用をサポートしています。

{{% alert %}}
W&B は公式の W&B Helm チャートでのインストールを推奨します。
{{% /alert %}}

#### 非特権ユーザーとしてコンテナを実行する

デフォルトでは、コンテナは `$UID` 999 を使用します。オーケストレーターが非 root ユーザーでの実行を求める場合は、`$UID` を 100000 以上、`$GID` を 0 に指定してください。

{{% alert %}}
ファイル システムのパーミッションを正しく機能させるため、W&B は root グループ（`$GID=0`）として起動する必要があります。
{{% /alert %}}

Kubernetes のセキュリティ コンテキスト例は次のようになります:

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## ネットワーキング

### ロード バランサー

適切なネットワーク境界でリクエストを終端するロード バランサーを稼働させてください。

一般的なロード バランサーの例:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](https://www.haproxy.org)

機械学習のペイロードを実行するすべてのマシンと、Web ブラウザでサービスにアクセスするデバイスが、このエンドポイントと通信できることを確認してください。

### SSL / TLS

W&B Server は SSL を終端しません。信頼できるネットワーク内部でも SSL 通信が必要というセキュリティ ポリシーの場合は、Istio や [サイドカー コンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/) などのツールの利用を検討してください。ロード バランサー自体は有効な証明書で SSL を終端する必要があります。自己署名証明書はサポートされず、ユーザーにさまざまな問題を引き起こします。可能であれば [Let's Encrypt](https://letsencrypt.org) のようなサービスを使って、ロード バランサーに信頼された証明書を提供するのが有効です。Caddy や Cloudflare のようなサービスは SSL の管理を代行してくれます。

### nginx 設定例

以下は、nginx をリバース プロキシとして使用する際の設定例です。

```nginx
events {}
http {
    # X-Forwarded-Proto を受け取った場合はその値を渡し、受け取っていない場合は
    # このサーバーへの接続に使われたスキームを渡す
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # また、上記の場合は HTTPS を強制する
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Host を受け取った場合はその値を渡し、受け取っていない場合は $http_host を渡す
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Port を受け取った場合はその値を渡し、受け取っていない場合は
    # クライアントが接続したサーバーのポートを渡す
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # Upgrade ヘッダーを受け取った場合は Connection を "upgrade" に設定し、
    # 受け取っていない場合はこのサーバーに渡された Connection ヘッダーを削除する
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

## インストールの検証

W&B Server が正しく設定されていることを確認します。ターミナル で次のコマンドを実行してください:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

起動時に W&B Server が出力するエラーを確認するには、ログ ファイルを確認します。次のコマンドを実行してください:

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