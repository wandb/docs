---
title: 온프레미스에 W&B 플랫폼 배포하기
description: 온프레미스 인프라에 W&B Server 호스팅
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-bare-metal
    parent: self-managed
weight: 5
---

{{% alert %}}
W&B는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})와 같은 완전 관리형 배포 옵션을 권장합니다. W&B의 완전 관리형 서비스는 매우 간단하고, 최소 또는 별도의 설정 없이 안전하게 사용할 수 있습니다.
{{% /alert %}}

관련 질문이 있다면 W&B 영업팀에 문의하세요: [contact@wandb.com](mailto:contact@wandb.com).

## 인프라 가이드라인

W&B를 배포하기 전에 [참고 아키텍처]({{< relref path="ref-arch.md#infrastructure-requirements" lang="ko" >}})의 인프라 요구사항을 반드시 확인하세요.

## MySQL 데이터베이스

{{% alert color="secondary" %}}
W&B는 MySQL 5.7 사용을 권장하지 않습니다. MySQL 5.7을 사용 중이라면, W&B Server의 최신 버전과 호환성을 최적화하려면 MySQL 8로 마이그레이션하는 것이 좋습니다. 현재 W&B Server는 `MySQL 8` 버전 중 `8.0.28` 이상만 지원합니다.
{{% /alert %}}

확장 가능한 MySQL 데이터베이스 운영을 더 쉽게 만들어주는 여러 엔터프라이즈 서비스가 있습니다. W&B는 다음 솔루션 중 하나의 사용을 권장합니다:

[Percona Server for MySQL](https://www.percona.com/software/mysql-database/percona-server)

[MySQL Operator for Kubernetes](https://github.com/mysql/mysql-operator)

W&B Server MySQL 8.0을 직접 운영하거나 MySQL 5.7에서 8.0으로 업그레이드할 경우, 아래 조건을 충족해야 합니다:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0의 `sort_buffer_size` 처리 방식 변경으로 인해 기본값 `262144`에서 `sort_buffer_size` 파라미터를 업데이트해야 할 수도 있습니다. MySQL이 W&B와 효율적으로 동작하려면 값을 `67108864`(64MiB)로 설정하는 것이 좋습니다. MySQL은 v8.0.28부터 이 설정을 지원합니다.

### 데이터베이스 고려사항

아래 SQL 쿼리를 사용해 데이터베이스와 사용자를 생성하세요. `SOME_PASSWORD`는 사용자가 원하는 비밀번호로 변경하세요.

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

{{% alert %}}
이 설정은 SSL 인증서가 신뢰된 경우에만 동작합니다. W&B는 자체 서명 인증서를 지원하지 않습니다.
{{% /alert %}}

### 파라미터 그룹 설정

데이터베이스 성능을 튜닝하려면 아래 파라미터 그룹이 모두 적용되어 있는지 확인하세요.

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## 오브젝트 스토리지

오브젝트 스토어는 외부에 [Minio 클러스터](https://min.io/docs/minio/kubernetes/upstream/index.html)로 운영하거나, 서명된 URL을 지원하는 Amazon S3 호환 오브젝트 스토어를 사용할 수 있습니다. 오브젝트 스토어가 서명된 URL을 지원하는지 확인하려면 [이 스크립트](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b)를 실행하세요.

또한, 아래의 CORS 정책을 오브젝트 스토어에 적용해야 합니다.

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

Amazon S3 호환 오브젝트 스토어에 연결할 때는 연결 문자열에 자격 증명을 직접 지정할 수 있습니다. 예시:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

오브젝트 스토어에 신뢰된 SSL 인증서를 설정하면 TLS 연결만 사용할 수 있도록 W&B에 요청할 수 있습니다. 이럴 경우 URL에 `tls` 쿼리 파라미터를 추가하면 됩니다. 예시는 다음과 같습니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

{{% alert color="secondary" %}}
이 설정은 SSL 인증서가 신뢰된 경우에만 동작합니다. W&B는 자체 서명 인증서를 지원하지 않습니다.
{{% /alert %}}

타사 오브젝트 스토어를 사용할 경우 `BUCKET_QUEUE`를 `internal://`로 설정하세요. 이렇게 하면 W&B 서버가 모든 오브젝트 알림을 외부 SQS 큐나 이와 유사한 시스템에 의존하지 않고 내부적으로 처리합니다.

자체 오브젝트 스토어를 운영할 때 가장 중요한 고려사항은 아래와 같습니다:

1. **저장 용량과 성능**: HDD(자기 디스크) 사용도 가능하지만, 디스크 용량을 반드시 모니터링해야 합니다. 일반적인 W&B 사용량은 수십 GB에서 수백 GB에 달합니다. 사용량이 많은 환경에서는 PB 단위 저장소가 필요할 수 있습니다.
2. **장애 허용성**: 최소한 오브젝트를 저장하는 물리 디스크는 RAID 어레이에 올리는 것이 좋습니다. minio를 사용하는 경우, [분산 모드](https://min.io/docs/minio/kubernetes/upstream/operations/concepts/availability-and-resiliency.html#distributed-minio-deployments)로 운영하는 것을 권장합니다.
3. **가용성**: 스토리지의 가용성을 보장하기 위해 모니터링을 설정해야 합니다.

자체 오브젝트 스토리지 서비스 대신 선택할 수 있는 엔터프라이즈 솔루션 예시는 아래와 같습니다:

1. [Amazon S3 on Outposts](https://aws.amazon.com/s3/outposts/)
2. [NetApp StorageGRID](https://www.netapp.com/data-storage/storagegrid/)

### MinIO 세팅

minio를 사용할 경우, 아래 코맨드를 실행해 버킷을 생성하세요.

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## W&B Server 애플리케이션을 Kubernetes에 배포

공식 W&B Helm 차트를 사용하는 것이 권장 설치 방법입니다. [Helm CLI 배포 가이드]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#deploy-wb-with-helm-cli" lang="ko" >}})를 따라 W&B Server 애플리케이션을 배포하세요.

### OpenShift

W&B는 [OpenShift Kubernetes 클러스터](https://www.redhat.com/en/technologies/cloud-computing/openshift) 환경에서 운영을 지원합니다.

{{% alert %}}
W&B는 공식 Helm 차트를 이용한 설치를 권장합니다.
{{% /alert %}}

#### 컨테이너를 비특권 사용자로 실행

기본적으로 컨테이너는 `$UID` 999를 사용합니다. 오케스트레이터에서 root가 아닌 사용자 실행이 요구될 경우 `$UID`를 100000 이상, `$GID`는 0으로 지정하세요.

{{% alert %}}
파일시스템 권한 동작을 위해 W&B는 반드시 root 그룹(`$GID=0`)으로 시작해야 합니다.
{{% /alert %}}

Kubernetes용 보안 컨텍스트 예시는 다음과 유사합니다:

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## 네트워킹

### 로드 밸런서

적절한 네트워크 경계에서 네트워크 요청을 멈추는 로드 밸런서를 실행하세요.

일반적으로 사용되는 로드 밸런서 예시:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](https://www.haproxy.org)

기계학습 작업을 실행하는 모든 머신과 브라우저에서 서비스를 이용하는 장치가 이 엔드포인트에 통신할 수 있어야 합니다.

### SSL / TLS

W&B Server는 SSL을 직접 처리하지 않습니다. 내부 신뢰 네트워크에서 SSL 통신이 필요한 보안 정책을 사용하는 경우, Istio와 [사이드카 컨테이너](https://istio.io/latest/docs/reference/config/networking/sidecar/) 같은 툴을 검토하세요. 로드 밸런서에서 신뢰된 인증서를 사용해 SSL을 종료하는 것이 좋습니다. 자체 서명 인증서 사용은 지원되지 않으며, 여러 사용자 문제를 초래할 수 있습니다. 가능하다면 [Let's Encrypt](https://letsencrypt.org)와 같은 서비스를 통해 인증서를 발급받는 것이 이상적입니다. Caddy, Cloudflare 등도 SSL을 자동 관리해줍니다.

### nginx 설정 예시

다음은 nginx를 reverse proxy로 사용할 때의 설정 예시입니다.

```nginx
events {}
http {
    # X-Forwarded-Proto 헤더를 받으면 그대로 전달, 아니면 서버 접속 scheme 사용
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # 위와 같이, HTTPS 강제 적용
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # X-Forwarded-Host 헤더를 받으면 그대로 전달, 아니면 $http_host 사용
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # X-Forwarded-Port 헤더를 받으면 그대로 전달, 아니면 클라이언트가 연결한 포트 사용
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # Upgrade 헤더가 있으면 "upgrade"로, 없으면 Connection 헤더 삭제
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

W&B Server가 정상적으로 설정되었는지 확인하세요. 터미널에서 다음 명령어를 실행하세요:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

시작 시 발생한 오류 확인을 위해 로그 파일을 검사할 수 있습니다. 아래 명령어를 실행하세요:

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

오류가 발생하면 W&B Support에 문의하세요.