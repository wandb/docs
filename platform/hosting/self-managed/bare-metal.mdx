---
title: Deploy W&B Platform On-premises
description: 온프레미스 인프라에 W&B Server 호스팅하기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-bare-metal
    parent: self-managed
weight: 5
---

{{% alert %}}
W&B에서는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 배포 유형과 같은 완전 관리형 배포 옵션을 권장합니다. W&B 완전 관리형 서비스는 사용하기 간편하고 안전하며, 최소한의 설정 또는 설정이 필요하지 않습니다.
{{% /alert %}}

관련 질문은 W&B Sales 팀에 문의하십시오: [contact@wandb.com](mailto:contact@wandb.com).

## 인프라 가이드라인

W&B 배포를 시작하기 전에 [참조 아키텍처]({{< relref path="ref-arch.md#infrastructure-requirements" lang="ko" >}}), 특히 인프라 요구 사항을 참조하십시오.

## MySQL 데이터베이스

{{% alert color="secondary" %}}
W&B는 MySQL 5.7 사용을 권장하지 않습니다. MySQL 5.7을 사용하는 경우 최신 버전의 W&B Server와의 최상의 호환성을 위해 MySQL 8로 마이그레이션하십시오. 현재 W&B Server는 `MySQL 8` 버전 `8.0.28` 이상만 지원합니다.
{{% /alert %}}

확장 가능한 MySQL 데이터베이스 운영을 간소화하는 여러 엔터프라이즈 서비스가 있습니다. W&B에서는 다음 솔루션 중 하나를 살펴보는 것이 좋습니다.

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

W&B Server MySQL 8.0을 실행하거나 MySQL 5.7에서 8.0으로 업그레이드하는 경우 아래 조건을 충족하십시오.

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0에서 `sort_buffer_size`를 처리하는 방식이 변경되었으므로 `sort_buffer_size` 파라미터를 기본값인 `262144`에서 업데이트해야 할 수 있습니다. W&B와 함께 MySQL이 효율적으로 작동하도록 값을 `67108864` (64MiB)로 설정하는 것이 좋습니다. MySQL은 v8.0.28부터 이 설정을 지원합니다.

### 데이터베이스 고려 사항

다음 SQL 쿼리를 사용하여 데이터베이스와 사용자를 만듭니다. `SOME_PASSWORD`를 원하는 비밀번호로 바꿉니다.

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

{{% alert %}}
이는 SSL 인증서를 신뢰할 수 있는 경우에만 작동합니다. W&B는 자체 서명된 인증서를 지원하지 않습니다.
{{% /alert %}}

### 파라미터 그룹 설정

데이터베이스 성능을 튜닝하려면 다음 파라미터 그룹이 설정되었는지 확인하십시오.

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## 오브젝트 스토리지
오브젝트 스토리지는 [Minio cluster](https://min.io/docs/minio/kubernetes/upstream/index.html) 또는 서명된 URL을 지원하는 Amazon S3 호환 오브젝트 스토리지에서 외부적으로 호스팅할 수 있습니다. [다음 스크립트](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b)를 실행하여 오브젝트 스토리지가 서명된 URL을 지원하는지 확인하십시오.

또한 다음 CORS 정책을 오브젝트 스토리지에 적용해야 합니다.

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

Amazon S3 호환 오브젝트 스토리지에 연결할 때 연결 문자열에 자격 증명을 지정할 수 있습니다. 예를 들어 다음과 같이 지정할 수 있습니다.

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

선택적으로 오브젝트 스토리지에 대해 신뢰할 수 있는 SSL 인증서를 구성한 경우 W&B에 TLS를 통해서만 연결하도록 지시할 수 있습니다. 이렇게 하려면 URL에 `tls` 쿼리 파라미터를 추가하십시오. 예를 들어 다음 URL 예제는 Amazon S3 URI에 TLS 쿼리 파라미터를 추가하는 방법을 보여줍니다.

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

{{% alert color="secondary" %}}
이는 SSL 인증서를 신뢰할 수 있는 경우에만 작동합니다. W&B는 자체 서명된 인증서를 지원하지 않습니다.
{{% /alert %}}

타사 오브젝트 스토리지를 사용하는 경우 `BUCKET_QUEUE`를 `internal://`로 설정하십시오. 이렇게 하면 W&B 서버가 외부 SQS 대기열 또는 이와 동등한 대기열에 의존하는 대신 모든 오브젝트 알림을 내부적으로 관리합니다.

자체 오브젝트 스토리지를 실행할 때 고려해야 할 가장 중요한 사항은 다음과 같습니다.

1. **저장 용량 및 성능**. 자기 디스크를 사용해도 괜찮지만 이러한 디스크의 용량을 모니터링해야 합니다. 평균적인 W&B 사용량은 10~100GB입니다. 사용량이 많으면 페타바이트의 스토리지 소비가 발생할 수 있습니다.
2. **결함 허용 오차**. 최소한 오브젝트를 저장하는 물리적 디스크는 RAID 어레이에 있어야 합니다. minio를 사용하는 경우 [분산 모드](https://min.io/docs/minio/kubernetes/upstream/operations/concepts/availability-and-resiliency.html#distributed-minio-deployments) 로 실행하는 것을 고려하십시오.
3. **가용성**. 스토리지를 사용할 수 있는지 확인하기 위해 모니터링을 구성해야 합니다.

자체 오브젝트 스토리지 서비스를 실행하는 대신 다음과 같은 여러 엔터프라이즈 대안이 있습니다.

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO 설정

minio를 사용하는 경우 다음 코맨드를 실행하여 버킷을 만들 수 있습니다.

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## Kubernetes에 W&B Server 애플리케이션 배포

권장되는 설치 방법은 공식 W&B Helm 차트를 사용하는 것입니다. [이 섹션]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#deploy-wb-with-helm-cli" lang="ko" >}})에 따라 W&B Server 애플리케이션을 배포하십시오.

### OpenShift

W&B는 [OpenShift Kubernetes cluster](https://www.redhat.com/en/technologies/cloud-computing/openshift) 내에서 운영하는 것을 지원합니다.

{{% alert %}}
공식 W&B Helm 차트를 사용하여 설치하는 것이 좋습니다.
{{% /alert %}}

#### 권한 없는 사용자로 컨테이너 실행

기본적으로 컨테이너는 `$UID` 999를 사용합니다. 오케스트레이터에서 루트가 아닌 사용자로 컨테이너를 실행해야 하는 경우 `$UID` >= 100000 및 `$GID` 0을 지정하십시오.

{{% alert %}}
파일 시스템 권한이 제대로 작동하려면 W&B가 루트 그룹(`$GID=0`)으로 시작해야 합니다.
{{% /alert %}}

Kubernetes에 대한 보안 컨텍스트 예제는 다음과 유사합니다.

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## 네트워킹

### 로드 밸런서

적절한 네트워크 경계에서 네트워크 요청을 중지하는 로드 밸런서를 실행합니다.

일반적인 로드 밸런서에는 다음이 포함됩니다.
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

기계 학습 페이로드를 실행하는 데 사용되는 모든 장치와 웹 브라우저를 통해 서비스에 액세스하는 데 사용되는 장치가 이 엔드포인트와 통신할 수 있는지 확인하십시오.

### SSL / TLS

W&B Server는 SSL을 중지하지 않습니다. 보안 정책에서 신뢰할 수 있는 네트워크 내에서 SSL 통신을 요구하는 경우 Istio와 같은 툴과 [side car containers](https://istio.io/latest/docs/reference/config/networking/sidecar/)를 사용하는 것이 좋습니다. 로드 밸런서 자체는 유효한 인증서로 SSL을 종료해야 합니다. 자체 서명된 인증서를 사용하는 것은 지원되지 않으며 사용자에게 여러 가지 문제가 발생할 수 있습니다. 가능하다면 [Let's Encrypt](https://letsencrypt.org)와 같은 서비스를 사용하여 로드 밸런서에 신뢰할 수 있는 인증서를 제공하는 것이 좋습니다. Caddy 및 Cloudflare와 같은 서비스는 SSL을 관리합니다.

### Nginx 구성 예제

다음은 nginx를 역방향 프록시로 사용하는 구성 예제입니다.

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

## 설치 확인

W&B Server가 올바르게 구성되었는지 확인하십시오. 터미널에서 다음 코맨드를 실행합니다.

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

로그 파일을 확인하여 W&B Server가 시작 시에 발생하는 오류를 확인하십시오. 다음 코맨드를 실행합니다.

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

오류가 발생하면 W&B 지원팀에 문의하십시오.
