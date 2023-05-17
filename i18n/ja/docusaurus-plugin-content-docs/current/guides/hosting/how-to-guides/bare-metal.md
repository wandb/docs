---
description: オンプレミスのベアメタルサーバーでW&Bサーバーをホスティングする方法
---

# オンプレミス / ベアメタル

W&Bサーバーを使って、スケーラブルな外部データストアに接続するベアメタルインフラストラクチャーを実行します。新しいインスタンスのプロビジョニング方法と外部データストアのプロビジョニングに関するガイダンスは以下を参照してください。

:::caution
W&Bアプリケーションのパフォーマンスは、運用チームが設定・管理するスケーラブルなデータストアに依存します。チームは、MySQL 5.7またはMySQL 8のデータベースサーバーと、AWS S3互換のオブジェクトストアを提供する必要があります。
:::

セールスチームに連絡するには、[contact@wandb.com](mailto:contact@wandb.com)にメールしてください。

### MySQLデータベース

:::caution
W&Bは現在、MySQL 5.7またはMySQL 8.0.28以降をサポートしています。
:::

#### MySQL 5.7

スケーラブルなMySQLデータベースを簡単に運用できるエンタープライズサービスがいくつかあります。以下のいずれかのソリューションを検討してみてください。

- [https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)
- [https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

#### MySQL 8.0

:::info
現在、Weights & Biasesアプリケーションは、`MySQL 8`のバージョン`8.0.28`以降のみをサポートしています。
:::
W&BサーバーをMySQL 8.0で実行する場合や、MySQL 5.7から8.0にアップグレードする場合には、追加のパフォーマンスチューニングが必要です。以下の設定でデータベースエンジンを調整することで、`wandb`アプリケーションのクエリパフォーマンスが全体的に向上します。

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0で`sort_buffer_size`の取り扱い方法が変更されたため、デフォルト値の`262144`から`sort_buffer_size`パラメータを更新する必要があります。データベースがwandbアプリケーションと効率的に動作するためには、値を`33554432(32MiB)`に設定することをお勧めします。ただし、これはMySQLのバージョン8.0.28以降でのみ有効です。

独自のMySQLデータベースを実行する際に考慮すべき最も重要な事項は以下の通りです。

1. **バックアップ**。データベースは定期的に別の施設にバックアップする必要があります。最低1週間の保持期間で、毎日バックアップを行うことをお勧めします。
2. **パフォーマンス**。サーバーが実行されているディスクは高速である必要があります。データベースはSSDまたはアクセラレーテッドNASで実行することをお勧めします。
3. **監視**。データベースは負荷のために監視されるべきです。CPU使用率が40％以上で5分以上続く場合、サーバーのリソースが不足している可能性が高いです。
4. **可用性**。可用性と耐久性の要件に応じて、リアルタイムでプライマリサーバーからすべての更新をストリーミングし、プライマリサーバーがクラッシュしたり破損したりした場合にフェイルオーバーできる別のマシンにホットスタンバイを設定することが望ましいです。

互換性のあるMySQLデータベースを用意したら、以下のSQLでデータベースとユーザーを作成できます（SOME_PASSWORDをお好みのパスワードに置き換えてください）。

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

### オブジェクトストア

オブジェクトストアは、外部でホストされた[Minioクラスター](https://docs.min.io/minio/k8s/)、または署名付きURLに対応した**S3互換オブジェクトストア**を使用できます。オブジェクトストアが署名付きURLに対応しているかどうかは、[次のスクリプト](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b)を実行して確認できます。S3互換オブジェクトストアに接続する際は、接続文字列で認証情報を指定できます。例えば、
```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

### Kubernetesの展開

以下のk8s yamlはカスタマイズすることができますが、Kubernetesでローカルを設定するための基本的な基盤として機能すべきです。

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
上記のk8s YAMLは、ほとんどのオンプレミス環境で動作するはずです。ただし、IngressおよびオプションのSSL終了の詳細は異なります。以下の[ネットワーキング](#networking)を参照してください。

### Helm Chart

W&BはHelm Chartを介して展開することもサポートしています。公式のW&B helm chartは[こちらから見つけることができます](https://github.com/wandb/helm-charts)。

### Openshift

W&Bは、[Openshift kubernetesクラスター](https://www.redhat.com/en/technologies/cloud-computing/openshift)内での運用をサポートしています。上記のkubernetesデプロイセクションの手順に従ってください。

#### 権限の無いユーザーとしてコンテナを実行する

デフォルトでは、コンテナは`$UID` 999で実行されます。 オーケストレータがルート以外のユーザーでコンテナを実行する必要がある場合は、`$UID`>= 100000および`$GID` 0を指定できます。ファイルシステムのアクセス許可が適切に機能するために、ルートグループ（`$GID=0`）として開始する必要があります。これは、Openshiftでコンテナを実行する場合のデフォルトの動作です。kubernetesのセキュリティコンテキストの例は次のようになります。

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

### Docker

_wandb/local_は、Dockerもインストールされているインスタンスで実行できます。 **8GB以上のRAMと4vCPUs**が必要です。以下のコマンドを実行してコンテナを起動します。

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
:::注意
このプロセスがクラッシュした場合に再起動されるように、プロセスマネージャを設定する必要があります。SystemDを使ってこれを行う方法についての概要は、[ここで見つける](https://blog.container-solutions.com/running-docker-containers-with-systemd)ことができます。
:::

## ネットワーキング

### ロードバランサー

適切なネットワーク境界でネットワークリクエストを終了させるロードバランサーを実行する必要があります。一部の顧客はwandbサービスをインターネット上で公開している一方、他の顧客は内部のVPN／VPC上で公開しています。機械学習のペイロードを実行するために使用されるマシンと、ウェブブラウザを介してサービスにアクセスするデバイスが、このエンドポイントに通信できるようにすることが重要です。一般的なロードバランサーには以下のようなものがあります。

1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

### SSL / TLS

W＆BサーバーはSSLを終了しません。信頼されるネットワーク内でのSSL通信がセキュリティポリシーで求められる場合は、Istioや[サイドカーコンテナ](https://istio.io/latest/docs/reference/config/networking/sidecar/)のようなツールを使用してください。ロードバランサー自体は、有効な証明書でSSLを終了する必要があります。自己署名証明書はサポートされておらず、ユーザーにとって多くの問題を引き起こします。可能であれば、[Let's Encrypt](https://letsencrypt.org)のようなサービスを使用して、ロードバランサーに信頼できる証明書を提供することが良い方法です。CaddyやCloudflareのようなサービスは、SSLを管理してくれます。

### Nginx設定の例

以下は、nginxをリバースプロキシとして使用する設定の例です。

```nginx
events {}
http {
    # If we receive X-Forwarded-Proto, pass it through; otherwise, pass along the
    # scheme used to connect to this server
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

# また上記のケースでは、HTTPSを強制する
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Hostが受信された場合は、それを渡す。それ以外の場合は、$http_hostを渡す
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Portが受信された場合は、それを渡す。それ以外の場合は、
    # クライアントが接続したサーバーポートを渡す
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # Upgradeが受信された場合は、Connectionを"upgrade"に設定する。それ以外の場合は、
    # このサーバーに渡された可能性があるConnectionヘッダーを削除する
    map $http_upgrade $proxy_connection {
        default upgrade;
        '' close;
    }

    server {
        listen 443 ssl;
        server_name         www.example.com;
        ssl_certificate     www.example.com.crt;
        ssl_certificate_key www.example.com.key;

```
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

サーバーがどのようにインストールされたかに関わらず、すべてが適切に設定されていることを確認することは良い考えです。W&Bは、CLIを使用してすべてが正しく設定されていることを簡単に確認できます。

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

エラーがある場合は、W&Bのサポートチームに連絡してください。また、アプリケーションが起動時に遭遇したエラーをログを確認することで確認できます。
### Docker

```bash
docker logs wandb-local
```

### Kubernetes

```bash
kubectl get pods

kubectl logs wandb-XXXXX-XXXXX
```