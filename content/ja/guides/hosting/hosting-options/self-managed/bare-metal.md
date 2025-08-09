---
title: W&B プラットフォームをオンプレミスでデプロイする
description: オンプレミス インフラストラクチャーで W&B Server をホスティング
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-bare-metal
    parent: self-managed
weight: 5
---

{{% alert %}}
W&B は、[W&B マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) などのフルマネージド型デプロイメントを推奨しています。W&B のフルマネージドサービスは、設定がほとんど不要、または最小限で使いやすく安全です。
{{% /alert %}}

関連するご質問があれば、W&B セールスチームまでご連絡ください: [contact@wandb.com](mailto:contact@wandb.com)。

## インフラストラクチャーのガイドライン

W&B のデプロイメントを開始する前に、[リファレンスアーキテクチャー]({{< relref path="ref-arch.md#infrastructure-requirements" lang="ja" >}})、特にインフラ要件をご確認ください。

## MySQL データベース

{{% alert color="secondary" %}}
W&B は MySQL 5.7 の利用を推奨していません。MySQL 5.7 をご利用の場合は、W&B Server の最新バージョンとの互換性を確保するために MySQL 8 への移行を推奨します。なお、W&B Server は `MySQL 8` の `8.0.28` 以上のみサポートしています。
{{% /alert %}}

スケーラブルな MySQL データベース運用を簡単にするエンタープライズ向けサービスがいくつかあります。W&B では以下のソリューションを検討することをおすすめしています。

[Percona Server for MySQL](https://www.percona.com/software/mysql-database/percona-server)

[MySQL Operator for Kubernetes](https://github.com/mysql/mysql-operator)

W&B Server で MySQL 8.0 を運用する場合、または MySQL 5.7 から 8.0 へアップグレードする場合は、以下の条件を満たしてください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0 で `sort_buffer_size` の扱いが一部変更されたため、デフォルト値である `262144` から `sort_buffer_size` パラメータを更新する必要がある場合があります。MySQL と W&B の効率的な連携のため、`67108864`（64MiB）への設定が推奨されています。MySQL のこの設定は v8.0.28 以降で利用可能です。

### データベースに関する注意点

以下の SQL クエリでデータベースおよびユーザーを作成してください。`SOME_PASSWORD` は任意のパスワードに置き換えてください。

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

{{% alert %}}
この方法は、SSL 証明書が信頼されている場合のみ動作します。W&B は自己署名証明書には対応していません。
{{% /alert %}}

### パラメーターグループの設定

データベースのパフォーマンス最適化のため、以下のパラメーターグループが設定されているかご確認ください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## オブジェクトストレージ
オブジェクトストアは、[Minio クラスター](https://min.io/docs/minio/kubernetes/upstream/index.html) や、署名付き URL をサポートする Amazon S3 互換のオブジェクトストアに外部ホストが可能です。お使いのオブジェクトストアが署名付き URL に対応しているかどうかは、[こちらのスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) を実行してご確認ください。

また、以下の CORS ポリシーをオブジェクトストアに適用する必要があります。

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

Amazon S3 互換オブジェクトストアに接続する際、コネクション文字列に認証情報を指定できます。例えば、以下のように指定します：

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

オプションとして、オブジェクトストアの SSL 証明書が信頼されている場合、TLS 経由のみ接続するよう W&B に指示できます。URL に `tls` クエリパラメータを追加してください。例えば下記のように Amazon S3 URI に TLS パラメータを追加します：

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

{{% alert color="secondary" %}}
この方法は、SSL 証明書が信頼されている場合のみ動作します。W&B は自己署名証明書には対応していません。
{{% /alert %}}

サードパーティ製オブジェクトストアを使用する場合は、`BUCKET_QUEUE` を `internal://` に設定してください。これにより、W&B サーバーが外部 SQS キューなどに頼らず、オブジェクト通知を内部で管理します。

独自のオブジェクトストアを運用する際に重要なポイントは以下の通りです。

1. **ストレージ容量とパフォーマンス。** 磁気ディスクの使用も可能ですが、その容量を常時監視してください。W&B の一般的な利用では数十 GB から数百 GB となります。大規模利用の場合は PB（ペタバイト）単位になることもあります。
2. **耐障害性。** 少なくとも、オブジェクトを格納する物理ディスクは RAID 構成を推奨します。minio 使用時は [分散モード](https://min.io/docs/minio/kubernetes/upstream/operations/concepts/availability-and-resiliency.html#distributed-minio-deployments) の利用もご検討ください。
3. **可用性。** ストレージが常に利用可能となるようモニタリング設定を行ってください。

独自のオブジェクトストレージサービス運用以外にも、下記のようなエンタープライズ向けオルタナティブがあります。

1. [Amazon S3 on Outposts](https://aws.amazon.com/s3/outposts/)
2. [NetApp StorageGRID](https://www.netapp.com/data-storage/storagegrid/)

### MinIO セットアップ

minio を利用する場合、以下のコマンドでバケットを作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## W&B Server アプリケーションを Kubernetes へデプロイ

推奨インストール方法は、公式 W&B Helm チャートの利用です。[Helm CLI でのデプロイ手順]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#deploy-wb-with-helm-cli" lang="ja" >}})をご参照のうえ、W&B Server アプリケーションをデプロイしてください。

### OpenShift

W&B は [OpenShift Kubernetes クラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift) での運用に対応しています。

{{% alert %}}
公式の W&B Helm チャートでのインストールを推奨します。
{{% /alert %}}

#### 非特権ユーザーでコンテナを起動する

デフォルトでは、コンテナは `$UID` 999 で起動されます。オーケストレーターから非 root ユーザーでの起動が要求される場合は、`$UID` >= 100000、`$GID` 0 を指定してください。

{{% alert %}}
ファイルシステムのパーミッションが正しく動作するために、W&B は root グループ（`$GID=0`）で起動する必要があります。
{{% /alert %}}

Kubernetes でのセキュリティコンテキスト例：

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## ネットワーキング

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを止めるロードバランサーを稼働させます。

よく使われるロードバランサー例：
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](https://www.haproxy.org)

機械学習の処理を行うすべてのマシンと、Web ブラウザ経由でサービスへアクセスするデバイスがこのエンドポイントと通信できることを確認してください。

### SSL / TLS

W&B Server は SSL を終端しません。セキュリティポリシー上、信頼できるネットワーク内で SSL 通信が必要な場合は、Istio などのツールと [サイドカーコンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/) の利用をご検討ください。ロードバランサー自体で信頼された証明書とともに SSL を終端する必要があります。自己署名証明書はサポートされておらず、ユーザーに様々な課題を引き起こします。[Let's Encrypt](https://letsencrypt.org) などのサービスを利用してロードバランサーに信頼された証明書を提供することを推奨します。Caddy や Cloudflare などのサービスは SSL の管理も行ってくれます。

### nginx 設定例

以下は nginx をリバースプロキシとして利用する場合の設定例です。

```nginx
events {}
http {
    # X-Forwarded-Proto を受信した場合はそれを利用、なければサーバー接続時のスキームをそのまま利用
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # 上記の場合、HTTPS を強制
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Host を受信した場合はそれを利用、なければ $http_host を利用
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Port を受信した場合はそれを利用、なければクライアント接続ポートを利用
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # Upgrade ヘッダーを受信した場合は Connection を "upgrade"、なければヘッダーを削除
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

## インストール確認

W&B Server が正しく設定されているか確認してください。ターミナルで以下のコマンドを実行します。

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

W&B Server 起動時のエラーを確認するには、ログファイルを確認してください。以下のコマンドを実行します。

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

エラーが発生した場合は、W&B サポートまでご連絡ください。