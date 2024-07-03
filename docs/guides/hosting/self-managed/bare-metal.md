---
title: Install on on-prem infra
description: オンプレミスインフラストラクチャー上での W&B サーバーのホスティング
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install on on-prem infra

:::info
W&B は、[W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) または [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) のデプロイメントタイプのような、完全管理されたデプロイメントオプションを推奨します。W&B の完全管理サービスは、簡単かつ安全に使用でき、ほとんどまたは全く設定を必要としません。
:::

Multi-tenant Cloud や Dedicated Cloud が組織に適していない場合、オンプレミスインフラストラクチャ上で W&B Server を実行することができます。

関連する質問があれば W&B の営業チームに連絡してください：[contact@wandb.com](mailto:contact@wandb.com)。

## インフラストラクチャのガイドライン

以下のインフラストラクチャのガイドラインセクションでは、アプリケーションサーバー、データベースサーバー、およびオブジェクトストレージの設定時に考慮すべき W&B の推奨事項を示しています。

:::tip
W&B は、W&B Kubernetes Operator を使用して W&B Server を Kubernetes クラスターにデプロイすることを強く推奨します。オペレーターを使用して Kubernetes クラスターにデプロイすることで、すべての既存および最新の W&B 機能を利用できます。
:::

:::caution
W&B アプリケーションのパフォーマンスは、運用チームが設定および管理するスケーラブルなデータストアに依存します。チームはアプリケーションが適切にスケーリングするために MySQL 8 データベースクラスターと AWS S3 互換のオブジェクトストアを提供する必要があります。
:::

### アプリケーションサーバー

最良のパフォーマンス、信頼性、および可用性を提供するために、以下の仕様を持つ 2 つのアベイラビリティゾーンノードグループとともに、W&B Server を独自の名前空間にデプロイすることをお勧めします：

| 仕様                          | 値                                  |
|-----------------------------|-----------------------------------|
| 帯域幅                       | デュアル 10 ギガビット+ Ethernet ネットワーク |
| ルートディスク帯域幅（Mbps） | 4,750+                            |
| ルートディスクプロビジョン（GB） | 100+                              |
| コア数                       | 4                                 |
| メモリ（GiB）                | 8                                 |

これにより、W&B Server はアプリケーションデータを処理し、一時ログを外部化する前に保存するための十分なディスクスペースを持つことが保証されます。また、高速で信頼性のあるデータ転送、スムーズな操作のための必要な処理能力とメモリを確保し、W&B が他のノイズの多い隣人の影響を受けないことが保証されます。

これらの仕様は最低要件であり、実際のリソースのニーズは W&B アプリケーションの特定の使用状況やワークロードに応じて異なる可能性があることに留意してください。アプリケーションのリソース使用量とパフォーマンスを監視し、必要に応じて調整することが重要です。

### データベースサーバー

W&B は、メタデータストアとして [MySQL 8](../self-managed/bare-metal.md#mysql-80) データベースを推奨します。ML 実践者のパラメータとメタデータの形はデータベースのパフォーマンスに大きな影響を与えます。データベースは通常、Practitioners がトレーニング Runs を追跡する際に段階的に書き込まれ、レポートとダッシュボードでクエリが実行される際には読み取りが多くなります。

最適なパフォーマンスを確保するために、以下のスタートスペックを持つサーバーに W&B データベースをデプロイすることをお勧めします：

| 仕様                          | 値                                  |
|------------------------------|-----------------------------------|
| 帯域幅                        | デュアル 10 ギガビット+ Ethernet ネットワーク |
| ルートディスク帯域幅（Mbps）  | 4,750+                            |
| ルートディスクプロビジョン（GB）| 1000+                             |
| コア数                        | 4                                 |
| メモリ（GiB）                 | 32                                |

同様に、データベースのリソース使用量とパフォーマンスを監視し、最適に動作することを確認し、必要に応じて調整することをお勧めします。

さらに、MySQL 8 用のデータベースをチューニングするために以下の[パラメータオーバーライド](../self-managed/bare-metal.md#mysql-80)をお勧めします。

### オブジェクトストレージ

W&B は、S3 API インターフェース、署名付き URL、および CORS をサポートするオブジェクトストレージと互換性があります。ストレージアレイを実践者の現在のニーズに合わせて仕様付け、定期的なキャパシティプランニングをお勧めします。

詳細なオブジェクトストアの設定は、[ハウツーセクション](../self-managed/bare-metal.md#object-store)に記載されています。

テスト済みで動作するプロバイダー：
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### Secure Storage Connector

[Secure Storage Connector](../data-security/secure-storage-connector.md) は、現時点ではベアメタルデプロイメントの Teams には利用できません。

## MySQL データベース

:::caution
W&B は MySQL 5.7 の使用を推奨しません。MySQL 5.7 を使用している場合、MySQL 8 に移行して最新バージョンの W&B Server と最高の互換性を確保してください。W&B Server は現在、`MySQL 8` バージョン `8.0.28` 以上のみをサポートしています。
:::

スケーラブルな MySQL データベースの運用を簡素化する多くのエンタープライズサービスがあります。以下のソリューションのいずれかを検討することをお勧めします：

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

W&B Server MySQL 8.0 を実行する場合、または MySQL 5.7 から 8.0 にアップグレードする場合、以下の条件を満たしてください：

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0 の `sort_buffer_size` の扱いにいくつかの変更があったため、`sort_buffer_size` パラメータをデフォルト値の `262144` から更新する必要があるかもしれません。データベースが W&B アプリケーションと効率的に動作するためには、値を `33554432(32MiB)` に設定することをお勧めします。この設定は MySQL バージョン 8.0.28 以上でのみ動作します。

### データベースに関する考慮事項

独自の MySQL データベースを運用する際には、以下の点を考慮してください：

1. **バックアップ**。 データベースを定期的に別の施設にバックアップする必要があります。毎日のバックアップと少なくとも 1 週間の保存を推奨します。
2. **パフォーマンス**。 サーバーが稼働しているディスクは高速である必要があります。SSD または加速された NAS 上でデータベースを実行することを推奨します。
3. **監視**。 データベースの負荷を監視する必要があります。システムの CPU 使用率が 5 分間以上 40% を超えて持続する場合、サーバーがリソース不足であることを示している可能性が高いです。
4. **可用性**。可用性と耐久性の要件に応じて、プライマリサーバーがクラッシュしたり破損した場合にフェイルオーバーとして使用できる別のマシンにホットスタンバイを設定し、すべての更新をリアルタイムでストリーミングすることを検討することができます。

次の SQL クエリでデータベースとユーザーを作成します。`SOME_PASSWORD` を任意のパスワードに置き換えてください：

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### パラメータグループ設定

データベースのパフォーマンスを調整するために、以下のパラメータグループが設定されていることを確認してください：

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 33554432
```

## オブジェクトストア
オブジェクトストアは、[Minio クラスター](https://docs.min.io/minio/k8s/) または署名付き URL をサポートする任意の Amazon S3 互換オブジェクトストアに外部ホストすることができます。次のスクリプトを実行して、オブジェクトストアが署名付き URL をサポートしているかを確認します：

[following script](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) 

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

Amazon S3 互換オブジェクトストアに接続するときに、接続文字列に資格情報を指定できます。例えば、以下のように指定できます：

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

任意で、オブジェクトストアに信頼できる SSL 証明書を設定することで、W&B に TLS を使用して接続するよう指示できます。そのためには、URL に `tls` クエリパラメータを追加します。次の URL 例は、Amazon S3 URI に TLS クエリパラメータを追加する方法を示しています：

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::caution
これは、SSL 証明書が信頼されている場合にのみ機能します。W&B は自己署名証明書をサポートしていません。
:::

サードパーティのオブジェクトストアを使用する場合は、`BUCKET_QUEUE` を `internal://` に設定します。これにより、外部の SQS キューや同等のものに依存せずに、W&B サーバーが内部でオブジェクト通知を管理するようになります。

独自のオブジェクトストアを運用する際に考慮すべき最も重要な点は次のとおりです：

1. **ストレージの容量とパフォーマンス**。 磁気ディスクを使用することは問題ありませんが、これらのディスクの容量を監視する必要があります。平均的な W&B の使用は 10 GB から 100 GB 以上に達します。ヘビーユーザーの場合、ペタバイトのストレージ消費に至ることもあります。
2. **フォルトトレランス**。 最低でも、オブジェクトを保存する物理ディスクは RAID アレイにする必要があります。minio を使用する場合、[分散モード](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed)で実行することを検討してください。
3. **可用性**。 ストレージが利用可能かどうかを確認するための監視を設定する必要があります。

独自のオブジェクトストレージサービスを運用する代わりになる多くのエンタープライズ代替案があります：

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO の設定

minio を使用する場合、次のコマンドを実行してバケットを作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## Kubernetes デプロイメント

以下の k8s yaml はカスタマイズ可能ですが、ローカルで Kubernetes を設定するための基本的な基盤として機能します。

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

上記の k8s YAML は、ほとんどのオンプレミスインストールで機能します。ただし、Ingress の詳細およびオプションの SSL 終端は異なる場合があります。以下の [ネットワーキング](#networking) を参照してください。

### Helm Chart

W&B は Helm Chart を使用したデプロイメントもサポートしています。公式の W&B helm chart は [こちら](https://github.com/wandb/helm-charts) にあります。

### Openshift

W&B は、[Openshift kubernetes クラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift) 内からの操作をサポートしています。上記の Kubernetes デプロイメントセクションの指示に従ってください。

#### 非特権ユーザーとしてコンテナを実行

デフォルトでは、コンテナは `$UID` 999 を使用します。オーケストレーターが非 root ユーザーでのコンテナ実行を要求する場合は、`$UID` ≥ 100000 および `$GID` 0 を指定します。

:::note
ファイルシステム権限が正常に機能するには、W&B は root グループ (`$GID=0`) として起動する必要があります。
:::

Kubernetes のセキュリティコンテキストの例は次のようになります：

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## ネットワーキング

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを終了するロードバランサーを実行します。

一般的なロードバランサーには以下が含まれます：
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

すべての機械学習ペイロードを実行するマシンと、ウェブブラウザを介してサービスにアクセスするデバイスがこのエンドポイントと通信できるようにしてください。

### SSL / TLS

W&B Server は SSL を終了しません。セキュリティポリシーで内部の信頼されたネットワーク内での SSL 通信が求められる場合は、Istio や [side car コンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/) などのツールの使用を検討してください。ロードバランサー自体は有効な証明書で SSL を終了する必要があります。自己署名証明書の使用はサポートされておらず、ユーザーに多くの課題をもたらします。可能な場合、[Let's Encrypt](https://letsencrypt.org) などのサービスを使用してロードバランサーに信頼された証明書を提供することは非常に良い方法です。Caddy や Cloudflare などのサービスは SSL を自動的に管理します。

### Nginx 設定例

以下はリバースプロキシとして Nginx を使用する例としての設定です。

```nginx
events {}
http {
    # X-Forwarded-Proto を受け取る場合はそれを渡し、それ以外の場合はこのサーバーへの接続時に使用されるスキームを渡す
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # 上記の場合、強制的に HTTPS を使用する
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Host を受け取る場合はそれを通し、それ以外の場合は $http_host を渡す
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Port を受け取る場合はそれを渡し、それ以外の場合はクライアントがサーバーに接続したポートを渡す
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # Upgrade を受け取る場合は接続を "upgrade" に設定し、それ以外の場合はこのサーバーに渡された Connection ヘッダーを削除する
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

W&B Server が適切に構成されているか確認します。ターミナルで次のコマンドを実行してください：

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

起動時に W&B Server がヒットするエラーを確認するには、次のコマンドを実行します。Docker または Kubernetes を使用するかに基づいてコマンドを実行してください：

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

エラーが発生した場合は W&B サポートに連絡してください。