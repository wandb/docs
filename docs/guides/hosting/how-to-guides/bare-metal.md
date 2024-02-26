---
description: Hosting W&B Server on baremetal servers on-premises
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# オンプレミス / ベアメタル

W&Bサーバーを使用して、スケーラブルな外部データストアに接続するベアメタルインフラストラクチャーを実行します。新しいインスタンスのプロビジョニング方法と外部データストアのプロビジョニングに関する指南については、以下を参照してください。

:::caution
W&Bアプリケーションのパフォーマンスは、運用チームが設定および管理する必要があるスケーラブルなデータストアに依存します。チームは、アプリケーションが適切にスケールするために、MySQL 5.7またはMySQL 8データベースサーバーとAWS S3互換のオブジェクトストアを提供する必要があります。
:::

W&Bセールスチームにお問い合わせください: [contact@wandb.com](mailto:contact@wandb.com)。

## インフラストラクチャーガイドライン

次のインフラストラクチャーガイドラインのセクションでは、アプリケーションサーバー、データベースサーバー、およびオブジェクトストレージを設定する際に考慮すべきW&Bの推奨事項を説明しています。

:::tip
W&BをKubernetesクラスターにデプロイすることをお勧めします。Kubernetesクラスターへのデプロイにより、すべてのW&B機能を使用し、[helmインターフェース](https://github.com/wandb/helm-charts)を利用できます。
:::

W&Bをベアメタルサーバーにインストールして手動で設定することもできます。ただし、W&Bサーバーは積極的に開発が進められており、一部の機能がK8sネイティブまたはカスタマーリソース定義に分割される可能性があります。その場合、スタンドアロンのDockerコンテナに特定の機能をバックポートすることはできません。

W&Bのオンプレミスインストールに関する質問がある場合は、W&Bサポート（support@wandb.com）までお問い合わせください。



### アプリケーションサーバー

最高のパフォーマンス、信頼性、可用性を提供するために、W&Bアプリケーションを独自の名前空間と以下の仕様を持つ2つのアベイラビリティーゾーンノードグループにデプロイすることをお勧めします。

| 仕様                        | 値                                  |
|--------------------------|-------------------------------------|
| 帯域幅                    | デュアル10ギガビット+イーサネットネットワーク |
| ルートディスク帯域幅（Mbps）  | 4,750+                              |
| ルートディスクプロビジョン（GB） | 100+                                |
| コア数                    | 4                                   |
| メモリー（GiB）             | 8                                   |

これにより、W&BはW&Bサーバーアプリケーションデータを処理し、外部化される前の一時的なログを保存するために十分なディスク容量が確保されます。また、高速かつ信頼性のあるデータ転送、スムーズな動作に必要な処理能力とメモリを確保し、W&Bがノイジーな隣人の影響を受けないことを保証します。

これらの仕様が最低要件であることに注意してください。実際のリソースの必要性は、W&Bアプリケーションの具体的な使用とワークロードによって異なる場合があります。アプリケーションのリソース使用量とパフォーマンスを監視して、最適に動作し、必要に応じて調整を行うことが重要です。

### データベースサーバー

W&Bは、メタデータストアとして[MySQL 8](../how-to-guides/bare-metal.md#mysql-80)データベースを推奨しています。MLエンジニアのパラメーターとメタデータの形状は、データベースのパフォーマンスに大きく影響します。データベースは、開発者がトレーニングrunをトラッキングする際に増分的に書き込まれ、レポートやダッシュボードでクエリが実行された際に読み込みが増えます。

最適なパフォーマンスを確保するために、W&Bデータベースを以下のスタートスペックのサーバーにデプロイすることを推奨します:

| 仕様                          | 値                                  |
|--------------------------- |------------------------------------|
| 帯域幅                       | デュアル10ギガビット+イーサネットネットワーク |
| ルートディスク帯域幅(Mbps)   | 4,750+                              |
| ルートディスクプロビジョン(GB) | 1000+                               |
| コア数                      | 4                                  |
| メモリー (GiB)              | 32                                 |


再度おすすめしますが、データベースのリソース使用量とパフォーマンスを監視し、最適に動作するように必要に応じて調整を行ってください。

また、MySQL 8のDBをチューニングするために、以下の[パラメータオーバーライド](../how-to-guides/bare-metal.md#mysql-80)をお勧めします。

### オブジェクトストレージ

W&Bは、S3 APIインターフェース、Signed URLs、CORSをサポートするオブジェクトストレージと互換性があります。開発者の現在のニーズに合わせてストレージ配列を設定し、定期的なペースで容量計画を立てることをお勧めします。

オブジェクトストアの設定に関する詳細は、[how-toセクション](../how-to-guides/bare-metal.md#object-store)で見つけることができます。

いくつかのテスト済みで動作するプロバイダー：
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### セキュアストレージコネクタ

[Secure Storage Connector](../secure-storage-connector.md)は、現在、ベアメタルデプロイメントのチームでは利用できません。

## MySQLデータベース

:::注意
W&Bは現在、MySQL 5.7またはMySQL 8.0.28以降をサポートしています。
:::

<Tabs
  defaultValue="apple"
  values={[
    {label: 'MySQL 5.7', value: 'apple'},
    {label: 'MySQL 8.0', value: 'orange'},
  ]}>
  <TabItem value="apple">
  
スケーラブルなMySQLデータベースを簡単に運用できるエンタープライズサービスがいくつかあります。以下のソリューションのいずれかを検討することをお勧めします。

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

  </TabItem>

  <TabItem value="orange">

:::info
W&Bアプリケーションは現在、`MySQL 8`のバージョン`8.0.28`およびそれ以降のバージョンのみをサポートしています。
:::

W&BサーバーMySQL 8.0を実行する場合、またはMySQL 5.7から8.0にアップグレードする場合は、以下の条件を満たしてください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0での`sort_buffer_size`の取り扱い方に変更があったため、デフォルト値の`262144`から`sort_buffer_size`パラメータを更新する必要があります。データベースがW＆Bアプリケーションで効率的に動作するように、値を`33554432(32MiB)`に設定することをお勧めします。ただし、これはMySQLバージョン8.0.28以上でのみ機能します。

  </TabItem>
</Tabs>

### データベースに関する注意事項

独自のMySQLデータベースを実行する場合は、以下を検討してください。

1. **バックアップ**. データベースは定期的に別の設備にバックアップする必要があります。1週間の保持期間を持つ毎日のバックアップを推奨します。
2. **パフォーマンス.** サーバーが実行されているディスクは高速である必要があります。データベースはSSDまたはアクセラレーションされたNASで実行することをお勧めします。
3. **監視.** データベースは負荷を監視する必要があります。CPU使用率がシステムの40％以上で5分以上継続される場合、サーバーがリソース不足である可能性が高いことを示しています。
4. **可用性.** 可用性と耐久性の要件に応じて、リアルタイムでプライマリサーバーからすべての更新をストリーミングし、プライマリサーバーがクラッシュしたり破損したりした場合にフェイルオーバーできる別のマシンにホットスタンバイを設定することがあります。

次のSQLクエリを使用してデータベースとユーザーを作成します。`SOME_PASSWORD` をお好みのパスワードに置き換えてください:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### パラメーターグループ設定

データベースのパフォーマンスをチューニングするために、以下のパラメーターグループを設定してください。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 33554432
```

## オブジェクトストア
オブジェクトストアは、外部で[Minioクラスター](https://docs.min.io/minio/k8s/)や、署名付きURLをサポートしているAmazon S3互換のオブジェクトストアでホストすることができます。オブジェクトストアが署名付きURLをサポートしているかどうかを確認するには、[こちらのスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b)を実行してください。

また、以下のCORSポリシーをオブジェクトストアに適用する必要があります。

```xml
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

Amazon S3互換のオブジェクトストアに接続する際に、接続文字列の中に認証情報を設定することができます。例えば、以下のように指定できます。

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

オブジェクトストア用に信頼できるSSL証明書を設定することで、W&BにTLS経由での接続のみを指定することができます。これを行うには、URLに`tls`クエリパラメータを追加します。以下のURL例では、Amazon S3 URIにTLSクエリパラメータを追加する方法を示しています。

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::注意
これは、SSL証明書が信頼されている場合にのみ機能します。W＆Bは自己署名証明書に対応していません。
:::

`BUCKET_QUEUE`を`internal://`に設定して、サードパーティのオブジェクトストアを使用する場合。これは、W＆Bサーバーに外部のSQSキューまたは同等のものに依存する代わりに、すべてのオブジェクト通知を内部で管理させるよう指示します。

独自のオブジェクトストアを実行する際に考慮すべき最も重要なことは次のとおりです。

1. **ストレージ容量とパフォーマンス**。磁気ディスクを使用しても問題ありませんが、これらのディスクの容量を監視する必要があります。平均的なW＆Bの使用は、数十から数百ギガバイトになります。大量の使用は、ペタバイトのストレージ消費になる可能性があります。
2. **フォールトトレランス**。少なくとも、オブジェクトを格納している物理ディスクはRAIDアレイ上にあるべきです。minioを使用する場合は、[分散モード](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed)で実行することを検討してください。
3. **可用性**。ストレージが利用可能であることを確認するために、監視を設定してください。

独自のオブジェクトストレージサービスを運用する代わりに、多くのエンタープライズ向けの代替手段があります。例えば：

1. [https://aws.amazon.com/jp/s3/outposts/](https://aws.amazon.com/jp/s3/outposts/)
2. [https://www.netapp.com/jp/data-storage/storagegrid/](https://www.netapp.com/jp/data-storage/storagegrid/)

### MinIOのセットアップ

MinIOを使用している場合は、以下のコマンドを実行してバケットを作成できます。

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## Kubernetes展開

次のk8s yamlはカスタマイズできますが、Kubernetesでローカルを設定するための基本的な基盤として機能するはずです。

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

```
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
            failureThreshold: 60 # マイグレーションに10分間許可
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

上記のk8s YAMLは、ほとんどのオンプレミス環境で動作するはずです。ただし、IngressおよびオプションのSSL終了の詳細は異なります。詳細は下記の[ネットワーキング](#networking)を参照してください。

### Helmチャート

W＆Bは、Helmチャートを介してデプロイすることもサポートしています。公式のW＆B Helmチャートは[こちらから見つけることができます](https://github.com/wandb/helm-charts)。

### Openshift

W＆Bは、[Openshift Kubernetesクラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift)内で動作することをサポートしています。上記のKubernetesデプロイメントセクションの手順に従ってください。

#### 特権のないユーザーとしてコンテナを実行する

デフォルトでは、コンテナは`$UID` 999を使用します。コンテナを非rootユーザーで実行する必要がある場合は、`$UID` >= 100000および`$GID` 0を指定します。

:::note
ファイルシステムの権限が正しく機能するために、W＆Bはルートグループ（`$GID=0`）として開始する必要があります。
:::

Kubernetesのセキュリティコンテキストの例は、次のようなものです。

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## Docker展開

_wandb/local_は、Dockerがインストールされているインスタンスでも実行できます。少なくとも8GBのRAMと4vCPUが望ましいです。

以下のコマンドを実行して、コンテナを起動します。

```bash
 docker run --rm -d \
   -e HOST=https://YOUR_DNS_NAME \
   -e LICENSE=XXXXX \
   -e BUCKET=s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME \
   -e BUCKET_QUEUE=internal:// \
   -e AWS_REGION=us-east1 \
   -e MYSQL=mysql://$USERNAME:$PASSWORD@$HOSTNAME/$DATABASE \
   -p 8080:8080 --name wandb-local wandb/local
```

:::caution
このプロセスがクラッシュした場合に再起動されるように、プロセスマネージャーを設定してください。SystemDを使用してこれを行う方法の良い概要は[こちら](https://blog.container-solutions.com/running-docker-containers-with-systemd)で見つかります。
:::

## ネットワーキング

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを終了するロードバランサーを実行します。

一般的なロードバランサーには以下のものがあります:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

機械学習のペイロードを実行するために使用されるすべてのマシン、およびウェブブラウザを介してサービスにアクセスするために使用されるデバイスが、このエンドポイントと通信できることを確認してください。

### SSL / TLS

W&BサーバーはSSLを終了しません。信頼できるネットワーク内でSSL通信が必要な場合は、Istioや[サイドカーコンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/)などのツールを利用してください。ロードバランサー自体が有効な証明書でSSLを終了させるべきです。自己署名証明書の使用はサポートされておらず、ユーザーに多くの問題を引き起こします。可能であれば、[Let's Encrypt](https://letsencrypt.org)のようなサービスを使って、ロードバランサーに信頼できる証明書を提供することが良い方法です。CaddyやCloudflareのようなサービスは、SSLを管理してくれます。

### Nginx設定の例

次に示すのは、Nginxをリバースプロキシとして使用する例の設定です。

```nginx
events {}
http {
    # X-Forwarded-Proto を受信した場合はそれを渡し、そうでない場合は、
    # このサーバーに接続するために使用されたスキームを渡す
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # 上記のケースでも、HTTPS を強制する
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Host を受信した場合はそれを渡し、そうでない場合は $http_host を渡す
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Port を受信した場合はそれを渡し、そうでない場合は、
    # クライアントが接続したサーバーポートを渡す
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

 # もしUpgradeが受信された場合、Connectionを"upgrade"に設定；そうでない場合、
    # このサーバーに渡されたかもしれないConnectionヘッダーを削除
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

W&Bサーバーが正しく設定されていることを確認します。ターミナルで以下のコマンドを実行してください：

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

W&Bサーバーが起動時にエラーが発生した場合は、ログファイルを確認してください。DockerまたはKubernetesを使用しているかどうかに基づいて、以下のコマンドを実行してください:

<Tabs
  defaultValue="apple"
  values={[
    {label: 'Apple', value: 'apple'},
    {label: 'Orange', value: 'orange'},
  ]}>
  <TabItem value="apple">

```bash
docker logs wandb-local
```

  </TabItem>
  <TabItem value="orange">

```bash
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

  </TabItem>
</Tabs>


問題が発生した場合は、W&Bサポートチームにお問い合わせください。