---
description: Hosting W&B Server on baremetal servers on-premises
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 온-프레미스 / 베어메탈

W&B 서버로 베어메탈 인프라를 실행하여 확장 가능한 외부 데이터 스토어에 연결합니다. 새 인스턴스를 프로비저닝하는 방법과 외부 데이터 스토어를 프로비저닝하는 가이드에 대한 안내는 다음을 참조하십시오.

:::caution
W&B 애플리케이션 성능은 운영 팀이 구성하고 관리해야 하는 확장 가능한 데이터 스토어에 따라 달라집니다. 팀은 애플리케이션이 제대로 확장되도록 MySQL 5.7 또는 MySQL 8 데이터베이스 서버와 AWS S3 호환 개체 스토어를 제공해야 합니다.
:::

W&B 영업팀에 문의: [contact@wandb.com](mailto:contact@wandb.com).

## 인프라 가이드라인

다음 인프라 가이드라인 섹션은 애플리케이션 서버, 데이터베이스 서버 및 개체 스토리지를 설정할 때 고려해야 할 W&B의 권장 사항을 개요합니다.

:::tip
W&B를 쿠버네티스 클러스터에 배포하는 것이 좋습니다. 쿠버네티스 클러스터에 배포하면 모든 W&B 기능을 사용할 수 있으며 [helm 인터페이스](https://github.com/wandb/helm-charts)를 사용할 수 있습니다.
:::

W&B를 베어메탈 서버에 설치하고 수동으로 구성할 수 있습니다. 그러나 W&B 서버는 활발하게 개발 중이며 특정 기능이 K8s 네이티브 또는 고객 리소스 정의로 분리될 수 있습니다. 이 경우 특정 기능을 독립형 Docker 컨테이너로 백포트 할 수 없습니다.

온-프레미스 설치 계획에 대해 질문이 있으면 W&B 지원팀에 문의하십시오. support@wandb.com.

### 애플리케이션 서버

최상의 성능, 신뢰성 및 가용성을 제공하려면 다음 사양을 갖춘 자체 네임스페이스 및 두 가용성 영역 노드 그룹에 W&B 애플리케이션을 배포하는 것이 좋습니다:

| 사양                        | 값                                |
|---------------------------|-----------------------------------|
| 대역폭                    | 듀얼 10 기가비트+ 이더넷 네트워크   |
| 루트 디스크 대역폭 (Mbps)   | 4,750+                            |
| 루트 디스크 프로비저닝 (GB) | 100+                              |
| 코어 수                   | 4                                 |
| 메모리 (GiB)              | 8                                 |

이는 W&B가 W&B 서버 애플리케이션 데이터를 처리하고 외부화되기 전에 임시 로그를 저장할 수 있는 충분한 디스크 공간을 확보하고, 빠르고 안정적인 데이터 전송을 보장하며, 원활한 운영에 필요한 처리 능력과 메모리를 확보하고, W&B가 다른 소음이 있는 이웃에 영향을 받지 않도록 보장합니다.

이 사양은 최소 요구 사항이며, W&B 애플리케이션의 특정 사용 및 작업 부하에 따라 실제 리소스 요구 사항이 달라질 수 있습니다. 애플리케이션이 최적으로 운영되고 필요에 따라 조정을 할 수 있도록 애플리케이션의 리소스 사용량과 성능을 모니터링하는 것이 중요합니다.

### 데이터베이스 서버

W&B는 메타데이터 저장소로 [MySQL 8](../how-to-guides/bare-metal.md#mysql-80) 데이터베이스를 권장합니다. ML 실무자의 파라미터와 메타데이터 구조는 데이터베이스의 성능에 큰 영향을 미칩니다. 데이터베이스는 일반적으로 실무자가 학습 실행을 추적할 때 점진적으로 작성되며, 리포트와 대시보드에서 쿼리가 실행될 때 더 많이 읽는 경향이 있습니다.

최적의 성능을 보장하기 위해 다음과 같은 시작 사양을 갖춘 서버에 W&B 데이터베이스를 배포하는 것이 좋습니다:

| 사양                       | 값                                 |
|---------------------------|-----------------------------------|
| 대역폭                    | 듀얼 10 기가비트+ 이더넷 네트워크    |
| 루트 디스크 대역폭 (Mbps)   | 4,750+                             |
| 루트 디스크 프로비저닝 (GB) | 1000+                              |
| 코어 수                   | 4                                 |
| 메모리 (GiB)              | 32                                |

다시 한번, 데이터베이스의 리소스 사용량과 성능을 모니터링하여 최적으로 운영되고 필요에 따라 조정하는 것이 좋습니다.

또한 MySQL 8용 DB를 조정하기 위해 다음 [파라미터 오버라이드](../how-to-guides/bare-metal.md#mysql-80)를 권장합니다.

### 개체 스토리지

W&B는 S3 API 인터페이스, Signed URLs 및 CORS를 지원하는 개체 스토리지와 호환됩니다. 실무자의 현재 요구 사항에 맞게 스토리지 어레이를 사양하고 정기적으로 용량 계획을 수립하는 것이 좋습니다.

개체 스토어 구성에 대한 자세한 내용은 [how-to 섹션](../how-to-guides/bare-metal.md#object-store)에서 찾을 수 있습니다.

테스트되고 작동하는 일부 제공업체:
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### 보안 스토리지 커넥터

베어메탈 배포의 경우 현재 팀에서 [보안 스토리지 커넥터](../secure-storage-connector.md)를 사용할 수 없습니다.

## MySQL 데이터베이스

:::caution
W&B는 현재 MySQL 5.7 또는 MySQL 8.0.28 이상을 지원합니다.
:::

<Tabs
  defaultValue="apple"
  values={[
    {label: 'MySQL 5.7', value: 'apple'},
    {label: 'MySQL 8.0', value: 'orange'},
  ]}>
  <TabItem value="apple">
  
확장 가능한 MySQL 데이터베이스를 운영하는 데 도움이 되는 몇 가지 엔터프라이즈 서비스가 있습니다. 다음 솔루션 중 하나를 살펴보는 것이 좋습니다:

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

  </TabItem>

  <TabItem value="orange">

:::info
W&B 애플리케이션은 현재 `MySQL 8` 버전 `8.0.28` 이상만 지원합니다.
:::

W&B 서버 MySQL 8.0을 실행하거나 MySQL 5.7에서 8.0으로 업그레이드할 때 아래 조건을 충족시키십시오:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0이 `sort_buffer_size`를 처리하는 방식에 일부 변경 사항이 있으므로 `sort_buffer_size` 파라미터를 기본값 `262144`에서 업데이트해야 할 수 있습니다. 데이터베이스가 W&B 애플리케이션과 효율적으로 작동하도록 값은 `33554432(32MiB)`로 설정하는 것이 좋습니다. 이는 MySQL 버전 8.0.28 이상에서만 작동합니다.
  
  </TabItem>
</Tabs>

### 데이터베이스 고려 사항

자체 MySQL 데이터베이스를 운영할 때 다음을 고려하십시오:

1. **백업**. 별도의 시설에 데이터베이스를 주기적으로 백업해야 합니다. 적어도 1주일의 보존 기간으로 매일 백업을 권장합니다.
2. **성능.** 서버가 실행 중인 디스크는 빨라야 합니다. SSD 또는 가속화된 NAS에서 데이터베이스를 실행하는 것이 좋습니다.
3. **모니터링.** 데이터베이스는 부하에 대해 모니터링되어야 합니다. 시스템의 > 40% CPU 사용량이 5분 이상 지속되면 서버가 리소스에 굶주린 좋은 징후일 수 있습니다.
4. **가용성.** 가용성 및 내구성 요구 사항에 따라 모든 업데이트를 실시간으로 스트리밍하는 별도의 기계에 핫 스탠바이를 구성하고 기본 서버가 충돌하거나 손상된 경우에 대비하여 장애 조치(failover)로 사용할 수 있도록 할 수 있습니다.

다음 SQL 쿼리를 사용하여 데이터베이스와 사용자를 생성하십시오. `SOME_PASSWORD`를 선택한 비밀번호로 대체하십시오:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### 파라미터 그룹 구성

다음 파라미터 그룹이 설정되어 데이터베이스 성능을 조정하는지 확인하십시오:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 33554432
```

## 개체 스토어
개체 스토어는 [Minio 클러스터](https://docs.min.io/minio/k8s/)에서 외부로 호스팅되거나 서명된 URL을 지원하는 Amazon S3 호환 개체 스토어에서 운영할 수 있습니다. 개체 스토어가 서명된 URL을 지원하는지 확인하는 [다음 스크립트](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b)를 실행하십시오.

또한 개체 스토어에 다음 CORS 정책을 적용해야 합니다.

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

Amazon S3 호환 개체 스토어에 연결할 때 자격 증명을 연결 문자열에 지정할 수 있습니다. 예를 들어, 다음을 지정할 수 있습니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

개체 스토어에 대한 신뢰할 수 있는 SSL 인증서를 구성한 경우 W&B에 TLS를 통해서만 연결하도록 지시할 수 있습니다. 이를 위해 URL에 `tls` 쿼리 파라미터를 추가하십시오. 예를 들어, 다음 URL 예시는 Amazon S3 URI에 TLS 쿼리 파라미터를 추가하는 방법을 보여줍니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::caution
이는 SSL 인증서가 신뢰할 수 있을 때만 작동합니다. W&B는 자체 서명된 인증서를 지원하지 않습니다.
:::

`BUCKET_QUEUE`를 `internal://`로 설정하면 서드파티 개체 스토어를 사용합니다. 이는 W&B 서버가 외부 SQS 큐 또는 동등한 것에 의존하지 않고 모든 개체 알림을 내부적으로 관리하도록 지시합니다.

자체 개체 스토어를 운영할 때 고려해야 할 가장 중요한 사항은 다음과 같습니다:

1. **저장 용량 및 성능**. 자기 디스크를 사용하는 것은 괜찮지만, 이러한 디스크의 용량을 모니터링해야 합니다. 평균 W&B 사용량은 수십에서 수백 기가바이트의 저장소를 생성합니다. 많은 사용량은 페타바이트의 저장소 소비를 초래할 수 있습니다.
2. **고장 내성.** 최소한, 개체를 저장하는 물리적 디스크는 RAID 어레이에 있어야 합니다. minio를 사용하는 경우 [분산 모드](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed)에서 실행하는 것을 고려하십시오.
3. **가용성.** 스토리지가 사용 가능한지 확인하기 위해 모니터링을 구성해야 합니다.

자체 개체 스토리지 서비스를 운영하는 대안은 많으며 다음과 같습니다:

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO 설정

minio를 사용하는 경우 다음 명령을 실행하여 버킷을 생성할 수 있습니다.

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## 쿠버네티스 배포

다음 k8s yaml은 커스터마이징할 수 있지만, 쿠버네티스에서 로컬을 구성하기 위한 기본적인 기반으로 사용될 수 있습니다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 

### 예제 Nginx 구성

다음은 nginx를 리버스 프록시로 사용하는 예제 구성입니다.

```nginx
events {}
http {
    # X-Forwarded-Proto를 받으면 통과시키고, 그렇지 않으면 이 서버에 연결하기 위해 사용된
    # 스킴을 통과시킵니다.
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # 위의 경우에도 HTTPS를 강제합니다
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Host를 받으면 통과시키고, 그렇지 않으면 $http_host를 통과시킵니다.
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Port를 받으면 통과시키고, 그렇지 않으면 클라이언트가 연결된
    # 서버 포트를 통과시킵니다.
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # Upgrade를 받으면 Connection을 "upgrade"로 설정하고, 그렇지 않으면 이 서버에 전달될 수 있는
    # 어떤 Connection 헤더도 삭제합니다.
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

## 설치 확인

W&B 서버가 제대로 구성되었는지 확인하세요. 터미널에서 다음 명령들을 실행하세요:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

W&B 서버가 시작할 때 발생한 어떤 오류도 확인하기 위해 로그 파일을 확인하세요. Docker나 Kubernetes를 사용하는지에 따라 다음 명령을 실행하세요:

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


오류가 발생하면 W&B 지원팀에 연락하세요.