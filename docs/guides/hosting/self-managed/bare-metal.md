---
title: Deploy W&B Platform On-premises
description: 온프레미스 인프라에 W&B 서버 호스팅
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info
W&B는 [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) 또는 [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) 배포 타입과 같은 완전 관리형 배포 옵션을 추천합니다. W&B의 완전 관리형 서비스는 간단하고 안전하게 사용할 수 있으며, 최소한의 설정만으로도 활용 가능합니다.
:::

Multi-tenant Cloud나 Dedicated Cloud가 조직에 적합하지 않은 경우, 온프레미스 인프라에서 W&B Server를 운영할 수 있습니다.

관련 질문이 있다면 W&B 영업팀에 문의하세요: [contact@wandb.com](mailto:contact@wandb.com).

## 인프라 가이드라인

다음 인프라 가이드라인 섹션은 애플리케이션 서버, 데이터베이스 서버, 오브젝트 스토리지를 설정할 때 고려해야 할 W&B 권장 사항을 설명합니다.

:::tip
W&B는 W&B Server를 Kubernetes 클러스터에서 W&B Kubernetes Operator를 사용하여 배포할 것을 강력히 권장합니다. 오퍼레이터를 사용하여 Kubernetes 클러스터에 배포하면 모든 기존 및 최신 W&B 기능을 사용할 수 있습니다.
:::

:::caution
W&B 애플리케이션 성능은 운영 팀이 구성하고 관리해야 하는 확장 가능한 데이터 저장소에 의존합니다. 팀은 애플리케이션이 제대로 확장되도록 MySQL 8 데이터베이스 클러스터와 AWS S3 호환 오브젝트 스토어를 제공해야 합니다.
:::

### 애플리케이션 서버

W&B는 성능, 신뢰성 및 가용성을 최적화하기 위해 W&B Server를 자체 네임스페이스와 두 개의 가용성 영역 노드 그룹으로 다음 사양을 사용하여 배포할 것을 권장합니다:

| 사양                        | 값                               |
|-----------------------------|-----------------------------------|
| 대역폭                      | 듀얼 10 기가비트+ 이더넷 네트워크 |
| 루트 디스크 대역폭 (Mbps)   | 4,750+                            |
| 루트 디스크 프로비저닝 (GB) | 100+                              |
| 코어 수                     | 4                                 |
| 메모리 (GiB)                | 8                                 |

이는 W&B Server가 애플리케이션 데이터를 처리하고 외부로 전송되기 전 임시 로그를 저장할 충분한 디스크 공간을 확보하는 것을 보장합니다.

또한, 빠르고 안정적인 데이터 전송, 매끄러운 운영을 위한 필요한 처리 능력과 메모리를 보장하며, W&B는 주변의 소음으로부터 영향을 받지 않게 됩니다.

이 사양은 최소 필요 요건이며, 실제 자원 필요량은 W&B 애플리케이션의 특정 사용 및 워크로드에 따라 다를 수 있다는 점을 유념해야 합니다. 애플리케이션의 자원 사용 및 성능을 모니터링하여 최적의 운영을 보장하고 필요에 따라 조정하는 것이 중요합니다.

### 데이터베이스 서버

W&B는 메타데이터 스토어로 [MySQL 8](#mysql-database) 데이터베이스를 권장합니다. 모델 파라미터의 형태와 관련 메타데이터가 데이터베이스의 성능에 영향을 미칩니다. 기계학습 실무자들이 더 많은 트레이닝 run을 추적함에 따라 데이터베이스 크기는 증가하고, run 테이블, 사용자 워크스페이스 및 리포트에서 쿼리가 실행될 때 무거운 읽기 부하를 발생시킵니다.

최적의 성능을 보장하기 위해 W&B는 W&B 데이터베이스를 다음 시작 사양을 가진 서버에 배포할 것을 권장합니다:

| 사양                        | 값                               |
|-----------------------------|-----------------------------------|
| 대역폭                      | 듀얼 10 기가비트+ 이더넷 네트워크 |
| 루트 디스크 대역폭 (Mbps)   | 4,750+                            |
| 루트 디스크 프로비저닝 (GB) | 1000+                             |
| 코어 수                     | 4                                 |
| 메모리 (GiB)                | 32                                |

다시 한 번, W&B는 데이터베이스의 자원 사용 및 성능을 모니터링하여 최적의 운영을 보장하고 필요에 따라 조정할 것을 권장합니다.

추가적으로, W&B는 MySQL 8을 위한 [파라미터 오버라이드](#mysql-database)를 활용하여 데이터베이스를 튜닝할 것을 권장합니다.

### 오브젝트 스토리지

W&B는 S3 API 인터페이스, 서명된 URL 및 CORS를 지원하는 오브젝트 스토리지를 호환합니다. W&B는 실무자의 현재 필요에 맞게 스토리지 배열을 지정하고, 정기적으로 용량을 계획할 것을 권장합니다.

오브젝트 스토어 설정에 대한 자세한 내용은 [사용 방법 섹션](../self-managed/bare-metal.md#object-store)에서 확인할 수 있습니다.

테스트 완료된 제공업체:
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### Secure Storage Connector

[SCS(Storage Connector Secure)](../data-security/secure-storage-connector.md)는 현재 베어 메탈 배포에서는 팀에게 제공되지 않습니다.

## MySQL 데이터베이스

:::caution
W&B는 MySQL 5.7의 사용을 권장하지 않습니다. MySQL 5.7을 사용 중이라면, W&B Server의 최신 버전과의 호환성을 위해 MySQL 8로 마이그레이션하세요. 현재 W&B Server는 `MySQL 8`의 `8.0.28` 이상 버전만 지원합니다.
:::

확장 가능한 MySQL 데이터베이스 운영을 더 쉽게 만드는 다양한 엔터프라이즈 서비스가 있습니다. W&B는 다음 솔루션 중 하나를 검토할 것을 권장합니다:

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

W&B Server MySQL 8.0을 실행하거나 MySQL 5.7에서 8.0으로 업그레이드할 경우 아래 조건을 충족하세요:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0이 `sort_buffer_size`를 처리하는 방식의 변경 사항으로 인해 기본 값인 `262144`에서 `sort_buffer_size` 파라미터를 업데이트해야 할 수도 있습니다. W&B 애플리케이션과 데이터베이스가 효율적으로 작동하기 위해 값은 `67108864(64MiB)`로 설정할 것을 추천합니다. 이는 MySQL 8.0.28 이상의 버전에서만 작동합니다.

### 데이터베이스 고려사항

자체 MySQL 데이터베이스를 운영할 때 다음 사항을 고려하세요:

1. **백업**. 데이터베이스를 별도의 시설에 정기적으로 백업해야 합니다. W&B는 최소 1주일 이상 보관하는 일일 백업을 권장합니다.
2. **성능**. 서버가 실행되는 디스크는 빠르게 구동되어야 합니다. W&B는 데이터베이스를 SSD 또는 가속 NAS에서 실행할 것을 권장합니다.
3. **모니터링**. 데이터베이스는 부하를 모니터링해야 합니다. CPU 사용률이 시스템의 40%를 초과하여 5분 이상 지속되면 서버가 리소스 부족 상태일 가능성이 높습니다.
4. **가용성**. 가용성 및 내구성 요구 사항에 따라, 모든 업데이트를 주 서버에서 실시간으로 스트리밍하여 핫 스탠바이를 별도의 머신에 구성하고, 주 서버 충돌 또는 손상 시 페일오버할 수 있도록 설정할 수 있습니다.

다음 SQL 쿼리로 데이터베이스와 사용자를 생성합니다. `SOME_PASSWORD`를 선택한 비밀번호로 교체하세요:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### 파라미터 그룹 설정

데이터베이스 성능을 튜닝하기 위해 다음 파라미터 그룹이 설정되어 있는지 확인합니다:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## 오브젝트 스토어
오브젝트 스토어는 [Minio 클러스터](https://docs.min.io/minio/k8s/)나 서명된 URL을 지원하는 Amazon S3 호환 오브젝트 스토어에 외부 호스팅될 수 있습니다. 오브젝트 스토어가 서명된 URL을 지원하는지 확인하기 위해 [다음 스크립트](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b)를 실행하세요.

또한, 다음 CORS 정책을 오브젝트 스토어에 적용해야 합니다.

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

Amazon S3 호환 오브젝트 스토어에 연결할 때 연결 문자열에서 자격 증명을 지정할 수 있습니다. 예를 들어, 다음과 같이 지정할 수 있습니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

선택적으로, 오브젝트 스토어에 대한 신뢰할 수 있는 SSL 인증서를 구성했다면 `tls` 쿼리 파라미터를 URL에 추가하여 W&B가 TLS를 통해서만 연결하도록 지시할 수 있습니다. 예를 들어, 다음 URL 예시는 Amazon S3 URI에 TLS 쿼리 파라미터를 추가하는 방법을 보여 줍니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::caution
이것은 SSL 인증서가 신뢰할 수 있는 경우에만 작동합니다. W&B는 자체 서명된 인증서를 지원하지 않습니다.
:::

제3자 오브젝트 스토어를 사용하는 경우 `BUCKET_QUEUE`를 `internal://`로 설정하세요. 이는 W&B 서버가 외부 SQS 큐나 유사한 곳에 의존하지 않고 모든 오브젝트 알림을 내부적으로 관리하도록 지시합니다.

자체 오브젝트 스토어를 운영하면서 고려해야 할 가장 중요한 사항은 다음과 같습니다:

1. **저장 용량 및 성능**. 자기 디스크를 사용하는 것이 좋지만, 이 디스크의 용량을 모니터링해야 합니다. 평균적인 W&B 사용은 10GB에서 100GB로 나타납니다. 대량 사용 시 수백 페타바이트의 저장소 소비가 발생할 수 있습니다.
2. **내결함성**. 최소한 오브젝트를 저장하는 물리적 디스크는 RAID 배열에 있어야 합니다. Minio를 사용하는 경우, [분산 모드](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed)에 실행하는 것을 고려하세요.
3. **가용성**. 스토리지가 사용 가능하도록 모니터링이 설정되어야 합니다.

자체 오브젝트 스토리지 서비스를 운영하는 것 외에도 다음과 같은 많은 기업용 대안이 있습니다:

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO 설정

Minio를 사용하는 경우 다음 코맨드를 실행하여 버킷을 생성할 수 있습니다.

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## W&B Server 애플리케이션을 Kubernetes에 배포

권장 설치 메소드는 공식 W&B Helm 차트를 사용하는 것입니다. W&B Server 애플리케이션을 배포하려면 [이 섹션](../operator#deploy-wb-with-helm-cli)을 따르세요.

### OpenShift

W&B는 [OpenShift Kubernetes 클러스터](https://www.redhat.com/en/technologies/cloud-computing/openshift) 내에서 운영을 지원합니다.

:::info
W&B는 공식 W&B Helm 차트 설치를 추천합니다.
:::
#### 비권한 사용자로 컨테이너 실행

기본적으로, 컨테이너는 `$UID`로 999를 사용합니다. 오케스트레이터가 비루트 사용자로 컨테이너를 실행해야 한다면 `$UID` >= 100000 및 `$GID` 0을 지정합니다.

:::note
W&B는 파일 시스템 권한이 적절히 작동하도록 루트 그룹(`$GID=0`)으로 시작해야 합니다.
:::

Kubernetes의 보안 컨텍스트 예시는 다음과 유사합니다:

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## 네트워킹

### 로드 밸런서

적절한 네트워크 경계를 설정하여 네트워크 요청을 중지하는 로드 밸런서를 실행합니다.

일반적인 로드 밸런서에는 다음이 포함됩니다:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

기계학습 페이로드를 실행하는 모든 머신과 웹 브라우저를 통해 서비스에 접근하는 디바이스는 이 엔드포인트와 통신할 수 있어야 합니다.

### SSL / TLS

W&B Server는 SSL을 중지하지 않습니다. 보안 정책 상 신뢰할 수 있는 네트워크 내에서 SSL 통신이 필요하다면 Istio 및 [사이드카 컨테이너](https://istio.io/latest/docs/reference/config/networking/sidecar/)와 같은 도구를 사용하는 것을 고려하세요. 로드 밸런서 자체는 유효한 인증서로 SSL을 종료해야 합니다. 자체 서명된 인증서를 사용하는 것은 지원되지 않으며 사용자에게 여러 가지 어려움을 초래할 것입니다. 가능한 경우 [Let's Encrypt](https://letsencrypt.org) 같은 서비스를 사용하여 로드 밸런서에 신뢰할 수 있는 인증서를 제공하는 것은 좋은 방법입니다. Caddy 및 Cloudflare와 같은 서비스는 SSL을 자동으로 관리해 줍니다.

### Nginx 설정 예시

다음은 Nginx를 리버스 프록시로 사용한 설정 예시입니다.

```nginx
events {}
http {
    # X-Forwarded-Proto를 수신하면 이를 통과; 그렇지 않으면 이 서버에 연결된 스킴을 전달합니다.
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # 위 경우에는 HTTPS를 강제합니다.
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Host를 수신하면 이를 통과; 그렇지 않으면 $http_host를 전달합니다.
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Port를 수신하면 이를 통과; 그렇지 않으면 클라이언트가 연결된 서버 포트를 전달합니다.
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # Upgrade를 수신하면 Connection을 "upgrade"로 설정; 그렇지 않으면 이 서버에 전달된 모든 Connection 헤더를 삭제합니다.
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

W&B Server가 적절히 구성되었는지 확인하세요. 터미널에서 다음 코맨드를 실행합니다:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

로그 파일을 확인하여 W&B Server가 시작할 때 발생한 오류가 있는지 확인하세요. Docker 또는 Kubernetes를 사용하는지에 따라 다음 코맨드를 실행하세요: 

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

W&B 지원 팀에 오류가 발생한 경우 연락하십시오.