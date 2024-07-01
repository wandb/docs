---
title: Install on on-prem infra
description: オンプレミス インフラストラクチャー上での W&B サーバー ホスティング
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install on on-prem infra

:::info
W&B は [W&B マルチテナントクラウド](../hosting-options/saas_cloud.md) または [W&B 専用クラウド](../hosting-options//dedicated_cloud.md) のデプロイタイプなど、フルマネージドのデプロイオプションを推奨します。W&B のフルマネージドサービスは、簡単で安全な使用が可能で、設定がほとんど不要です。
:::

マルチテナントクラウドや専用クラウドが組織に適していない場合、オンプレミスのインフラストラクチャ上で W&B Server を実行することができます。

関連する質問がある場合は、W&B セールスチームにお問い合わせください：[contact@wandb.com](mailto:contact@wandb.com)。

## インフラストラクチャガイドライン

以下のインフラストラクチャガイドラインセクションでは、アプリケーションサーバー、データベースサーバー、およびオブジェクトストレージの設定時に考慮すべき W&B の推奨事項を概説します。

:::tip
W&B は W&B Kubernetes Operator を使用して W&B Server を Kubernetes クラスターにデプロイすることを強く推奨します。オペレーターと一緒に Kubernetes クラスターにデプロイすることで、すべての既存および最新の W&B 機能を利用できるようになります。
:::

:::caution
W&B アプリケーションのパフォーマンスは、オペレーションチームが設定および管理するスケーラブルなデータストアに依存します。チームは、MySQL 8 データベースクラスターと AWS S3 互換のオブジェクトストアを提供する必要があります。
:::

### アプリケーションサーバー

最良のパフォーマンス、信頼性、および可用性を提供するために、専用の名前空間と以下の仕様を持つ二重可用性ゾーンノードグループに W&B Server をデプロイすることをお勧めします。

| Specification              | Value                             |
|----------------------------|-----------------------------------|
| Bandwidth                  | Dual 10 Gigabit+ Ethernet Network |
| Root Disk Bandwidth (Mbps) | 4,750+                            |
| Root Disk Provision (GB)   | 100+                              |
| Core Count                 | 4                                 |
| Memory (GiB)               | 8                                 |

これにより、W&B Server はアプリケーションデータを処理し、外部化される前に一時ログを保存するための十分なディスク容量を確保します。また、高速かつ信頼性の高いデータ転送、スムーズな操作のための必要な処理能力とメモリ、およびノイズの多い近隣からの影響を受けないようにします。

これらの仕様は最低要件であり、実際のリソースニーズは W&B アプリケーションの特定の使用状況やワークロードに応じて異なる可能性があることを念頭に置いてください。アプリケーションのリソース使用状況とパフォーマンスを監視し、必要に応じて調整することが重要です。

### データベースサーバー

W&B は [MySQL 8](../self-managed/bare-metal.md#mysql-80) データベースをメタデータストアとして推奨します。ML プラクティショナーのパラメータとメタデータの形状はデータベースのパフォーマンスに大きな影響を与えます。データベースは通常、プラクティショナーがトレーニング run をトラッキングする際にインクリメンタルに書き込まれ、レポートやダッシュボードでクエリが実行されるときは読み込みが多くなります。

最適なパフォーマンスを確保するために、以下の初期仕様を持つサーバーに W&B データベースをデプロイすることをお勧めします。

| Specification              | Value                             |
|--------------------------- |-----------------------------------|
| Bandwidth                  | Dual 10 Gigabit+ Ethernet Network |
| Root Disk Bandwidth (Mbps) | 4,750+                            |
| Root Disk Provision (GB)   | 1000+                              |
| Core Count                 | 4                                 |
| Memory (GiB)               | 32                                |

また、データベースのリソース使用状況とパフォーマンスを監視し、最適に動作するようにし、必要に応じて調整することをお勧めします。

さらに、MySQL 8 用のデータベースをチューニングするための [パラメータオーバーライド](../self-managed/bare-metal.md#mysql-80) をお勧めします。

### オブジェクトストレージ

W&B は S3 API インターフェース、サインド URL、および CORS をサポートするオブジェクトストレージと互換性があります。プラクティショナーの現在のニーズに合わせてストレージアレイを特定し、定期的に容量計画を行うことをお勧めします。

オブジェクトストアの設定に関する詳細は、[ハウツーセクション](../self-managed/bare-metal.md#object-store) にあります。

テスト済みで動作するプロバイダ:
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### セキュアストレージコネクタ

ベアメタルデプロイメントの場合、現時点でチームにセキュアストレージコネクタは利用できません。

## MySQL データベース

:::caution
W&B は MySQL 5.7 の使用を推奨しません。MySQL 5.7 を使用している場合は、W&B Server の最新バージョンとの互換性を確保するために MySQL 8 に移行してください。W&B Server は現在 `MySQL 8` バージョン `8.0.28` 以降のみをサポートしています。
:::

スケーラブルな MySQL データベースの操作を容易にするいくつかのエンタープライズサービスがあります。以下のソリューションのいずれかを検討することをお勧めします。

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

W&B Server MySQL 8.0 を実行する場合、または MySQL 5.7 から 8.0 へアップグレードする場合、以下の条件を満たしてください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0 の `sort_buffer_size` の処理方法のいくつかの変更により、デフォルト値 `262144` から `sort_buffer_size` パラメータを更新する必要があるかもしれません。データベースが W&B アプリケーションと効率的に連携するために、値を `33554432（32MiB）` に設定することをお勧めします。これは MySQL バージョン 8.0.28 以降でのみ機能します。

### データベースの考慮事項

独自の MySQL データベースを運用する際には、以下の点を考慮してください。

1. **バックアップ**. データベースを別の施設に定期的にバックアップする必要があります。少なくとも 1 週間の保持期間で毎日のバックアップをお勧めします。
2. **パフォーマンス**. サーバーが実行されているディスクは高速であるべきです。データベースを SSD または高速化された NAS 上で実行することをお勧めします。
3. **モニタリング**. データベースは負荷を監視する必要があります。CPU 使用率が 5 分以上 40％ を超えて持続する場合、それはサーバーがリソース不足である可能性があります。
4. **可用性**. 可用性と耐久性の要件に応じて、プライマリサーバーがクラッシュしたり破損した場合にフェイルオーバーできるように、すべての更新をリアルタイムでストリームするホットスタンバイを別のマシンに設定することを検討してください。

次の SQL クエリを使用してデータベースとユーザーを作成します。`SOME_PASSWORD` を選択したパスワードに置き換えてください。

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### パラメータグループの設定

データベースパフォーマンスを調整するために、以下のパラメータグループが設定されていることを確認してください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 33554432
```

## オブジェクトストア

オブジェクトストアは、[Minio クラスター](https://docs.min.io/minio/k8s/) 上またはサインド URL をサポートする Amazon S3 互換のオブジェクトストア上に外部ホストすることができます。オブジェクトストアがサインド URL をサポートしているかどうかを確認するために [次のスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) を実行してください。

さらに、オブジェクトストアに次の CORS ポリシーを適用する必要があります。

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

Amazon S3 互換のオブジェクトストアに接続する際に、接続文字列に資格情報を指定できます。たとえば、次のように指定できます。

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

オブジェクトストアのトラステッド SSL 証明書を設定する場合に限り、TLS 経由でのみ接続するように W&B に指示することもできます。これを行うには、`tls` クエリパラメータを URL に追加します。以下の URL 例は、Amazon S3 URI に TLS クエリパラメータを追加する方法を示しています。

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::caution
これは、SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしていません。
:::

サードパーティのオブジェクトストアを使用する場合は、`BUCKET_QUEUE` を `internal://` に設定してください。これにより、外部の SQS キューや同等のものに依存せずに、すべてのオブジェクト通知を内部で管理するように W&B サーバーに指示します。

自身のオブジェクトストアを運用する際に考慮すべき最も重要な点は以下のとおりです。

1. **ストレージキャパシティとパフォーマンス**。磁気ディスクの使用は問題ありませんが、これらのディスクの容量を監視する必要があります。平均的な W&B の使用では数十から数百のギガバイトです。重い使用ではペタバイトのストレージ消費になります。
2. **フォールトトレランス**。最低でも、オブジェクトを保存する物理ディスクは RAID 配列上にあるべきです。Minio を使用する場合は、[分散モード](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed) で実行することを検討してください。
3. **可用性**。ストレージが利用可能であることを確認するために監視を設定してください。

自身のオブジェクトストレージサービスを運用する以外にも、多くのエンタープライズオルタナティブがあります。

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO セットアップ

Minio を使用する場合、次のコマンドを実行してバケットを作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## Kubernetes デプロイメント

以下の k8s yaml はカスタマイズできますが、Kubernetes 上でローカルに設定するための基本的な基盤として役立ちます。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wandb
  labels:
    app: wandb
spec:
  strategy:
    type: RollingUpdate
  replicas: 1
  selector:
    matchLabels:
      app: wandb
  template:
    metadata:
      labels:
        app: wandb
    spec:
      containers:
        - name: wandb
          env:
            - name: HOST
              value: https://YOUR_DNS_NAME
            - name: LICENSE
              value: XXXXXXXXXXXXXXX
            - name: BUCKET
              value: s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
            - name: BUCKET_QUEUE
              value: internal://
            - name: AWS_REGION
              value: us-east-1
            - name: MYSQL
              value: mysql://$USERNAME:$PASSWORD@$HOSTNAME/$DATABASE
          imagePullPolicy: IfNotPresent
          image: wandb/local:latest
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
          readinessProbe:
            httpGet:
              path: /ready
              port: http
          startupProbe:
            httpGet:
              path: /ready
              port: http
            failureThreshold: 60 # allow 10 minutes for migrations
          resources:
            requests:
              cpu: '2000m'
              memory: 4G
            limits:
              cpu: '4000m'
              memory: 8G
---
apiVersion: v1
kind: Service
metadata:
  name: wandb-service
spec:
  type: NodePort
  selector:
    app: wandb
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: wandb-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  defaultBackend:
    service:
      name: wandb-service
      port:
        number: 80
```

上記の k8s YAML はほとんどのオンプレミスインストールで機能するはずです。ただし、Ingress の詳細やオプションの SSL 終端は異なる場合があります。以下の [ネットワーキング](#networking) を参照してください。

### Helm Chart

W&B は Helm Chart を使用したデプロイメントもサポートしています。公式の W&B Helm チャートは[こちら](https://github.com/wandb/helm-charts)です。

### Openshift

W&B は [Openshift Kubernetes クラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift) 内からの操作をサポートしています。上記の Kubernetes デプロイメントセクションの指示に従うだけです。

#### コンテナを非特権ユーザーとして実行する

デフォルトでは、コンテナは `$UID` 999 を使用します。オーケストレーターが非ルートユーザーでコンテナを実行する必要がある場合、`$UID` >= 100000 および `$GID` 0 を指定します。

:::note
ファイルシステムの権限が正常に機能するためには、W&B は root グループ (`$GID=0`) で起動する必要があります。
:::

Kubernetes のセキュリティコンテキストの例は次のようになります。

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## ネットワーキング

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを終端するロードバランサーを実行します。

一般的なロードバランサーには次のものがあります。
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

機械学習のペイロードを実行するために使用されるすべてのマシンと、Web ブラウザを通じてサービスにアクセスするために使用されるデバイスがこのエンドポイントに通信できることを確認してください。

### SSL / TLS

W&B Server は SSL 終端を行いません。セキュリティポリシーで信頼されたネットワーク内での SSL 通信が求められる場合、Istio や [side car コンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/) などのツールを使用することを検討してください。ロードバランサー自体は、有効な証明書で SSL を終端するべきです。自己署名証明書の使用はサポートされておらず、ユーザーにとって多くの課題を引き起こすことになります。可能であれば、[Let's Encrypt](https://letsencrypt.org) などのサービスを使用してロードバランサーに信頼された証明書を提供することを強くお勧めします。Caddy や Cloudflare などのサービスは SSL 管理を代行します。

### Nginx 設定の例

以下は nginx をリバースプロキシとして使用する場合の設定例です。

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

W&B Server が正しく構成されているかを確認します。次のコマンドをターミナルで実行してください。

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

起動時に W&B Server がエラーを示しているかどうかログファイルを確認します。Docker または Kubernetes を使用しているかに応じて、次のコマンドを実行します。

<Tabs
  defaultValue="docker"
  values={[
    {label: 'Docker', value: 'docker'},
    {label: 'Kubernetes', value: 'kubernetes'},
  ]}>
  <TabItem value="docker">

```bash
docker logs wandb-local
```

  </TabItem>
  <TabItem value="kubernetes">

```bash
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

  </TabItem>
</Tabs>

