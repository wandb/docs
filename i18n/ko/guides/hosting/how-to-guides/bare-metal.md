---
description: Hosting W&B Server on baremetal servers on-premises
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 온프레미스 / 베어메탈

W&B 서버와 연결되어 확장 가능한 외부 데이터 스토어를 사용하는 베어메탈 인프라를 운영하세요. 새 인스턴스를 프로비저닝하는 방법과 외부 데이터 스토어를 프로비저닝하는 안내에 대해서는 다음을 참조하세요.

:::caution
W&B 애플리케이션 성능은 운영 팀이 구성하고 관리해야 하는 확장 가능한 데이터 스토어에 따라 달라집니다. 팀은 애플리케이션이 제대로 확장되도록 MySQL 5.7 또는 MySQL 8 데이터베이스 서버와 AWS S3 호환 오브젝트 스토어를 제공해야 합니다.
:::

W&B 영업팀에 문의하기: [contact@wandb.com](mailto:contact@wandb.com).

## 인프라 가이드라인

다음 인프라 가이드라인 섹션은 애플리케이션 서버, 데이터베이스 서버 및 오브젝트 스토리지를 설정할 때 고려해야 할 W&B의 권장 사항을 개요합니다.

:::tip
W&B를 Kubernetes 클러스터에 배포하는 것이 좋습니다. Kubernetes 클러스터에 배포하면 모든 W&B 기능을 사용하고 [helm 인터페이스](https://github.com/wandb/helm-charts)를 사용할 수 있습니다.
:::

W&B를 베어메탈 서버에 설치하고 수동으로 구성할 수 있습니다. 그러나 W&B 서버는 활발하게 개발 중이며 일부 기능은 K8s 네이티브 또는 사용자 리소스 정의로 분할될 수 있습니다. 이 경우, 일부 기능을 독립 실행형 Docker 컨테이너로 다시 가져올 수 없습니다.

온프레미스 설치 계획 및 W&B에 대해 질문이 있는 경우 support@wandb.com으로 W&B 지원팀에 문의하세요.

### 애플리케이션 서버

최상의 성능, 신뢰성 및 가용성을 제공하기 위해 다음 사양을 갖춘 자체 네임스페이스 및 두 가용성 영역 노드 그룹에 W&B 애플리케이션을 배포하는 것이 좋습니다:

| 사양                         | 값                                  |
|---------------------------|------------------------------------|
| 대역폭                       | 듀얼 10 기가비트+ 이더넷 네트워크        |
| 루트 디스크 대역폭 (Mbps)      | 4,750+                             |
| 루트 디스크 프로비저닝 (GB)   | 100+                               |
| 코어 수                     | 4                                  |
| 메모리 (GiB)                | 8                                  |

이것은 W&B가 W&B 서버 애플리케이션 데이터를 처리하고 외부화되기 전에 임시 로그를 저장할 수 있는 충분한 디스크 공간을 확보하고, 빠르고 안정적인 데이터 전송을 보장하며, 원활한 작동에 필요한 처리 성능과 메모리를 보장하고, W&B가 다른 작업에 영향을 받지 않도록 합니다.

이러한 사양은 최소 요구 사항이며, 실제 리소스 요구 사항은 W&B 애플리케이션의 특정 사용 및 워크로드에 따라 달라질 수 있습니다. 애플리케이션의 리소스 사용량과 성능을 모니터링하여 최적으로 운영되도록 하고 필요에 따라 조정하는 것이 중요합니다.

### 데이터베이스 서버

W&B는 메타데이터 저장소로 [MySQL 8](../how-to-guides/bare-metal.md#mysql-80) 데이터베이스를 권장합니다. ML 실무자의 파라미터와 메타데이터의 구조는 데이터베이스의 성능에 큰 영향을 미칩니다. 데이터베이스는 일반적으로 실무자가 트레이닝 실행을 추적할 때 점진적으로 작성되며, 리포트 및 대시보드에서 쿼리를 실행할 때 읽기 중심이 됩니다.

최적의 성능을 보장하기 위해 다음 시작 사양을 갖춘 서버에 W&B 데이터베이스를 배포하는 것이 좋습니다:

| 사양                         | 값                                  |
|----------------------------|-----------------------------------|
| 대역폭                       | 듀얼 10 기가비트+ 이더넷 네트워크        |
| 루트 디스크 대역폭 (Mbps)      | 4,750+                             |
| 루트 디스크 프로비저닝 (GB)   | 1000+                              |
| 코어 수                     | 4                                  |
| 메모리 (GiB)                | 32                                 |

다시 한번, 데이터베이스의 리소스 사용량과 성능을 모니터링하여 최적으로 운영되도록 하고 필요에 따라 조정하는 것이 좋습니다.

또한, MySQL 8을 위한 DB를 튜닝하기 위해 다음 [파라미터 오버라이드](../how-to-guides/bare-metal.md#mysql-80)를 권장합니다.

### 오브젝트 스토리지

W&B는 S3 API 인터페이스, Signed URLs 및 CORS를 지원하는 오브젝트 스토리지와 호환됩니다. 실무자의 현재 요구 사항에 맞게 스토리지 어레이를 스펙하고 정기적으로 용량 계획을 수립하는 것이 좋습니다.

오브젝트 스토어 구성에 대한 자세한 내용은 [how-to 섹션](../how-to-guides/bare-metal.md#object-store)에서 찾을 수 있습니다.

테스트되고 작동하는 몇 가지 제공업체:
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### 보안 스토리지 커넥터

베어메탈 배포에 대해 현재 팀에서 [보안 스토리지 커넥터](../secure-storage-connector.md)를 사용할 수 없습니다.

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
  
확장 가능한 MySQL 데이터베이스를 운영하는 것을 단순화하는 여러 엔터프라이즈 서비스가 있습니다. 다음 해결책 중 하나를 살펴보는 것이 좋습니다:

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

  </TabItem>

  <TabItem value="orange">

:::info
W&B 애플리케이션은 현재 `MySQL 8` 버전 `8.0.28` 이상만 지원합니다.
:::

W&B 서버 MySQL 8.0을 실행하거나 MySQL 5.7에서 8.0으로 업그레이드할 때 아래 조건을 충족하세요:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

MySQL 8.0이 `sort_buffer_size`를 처리하는 방식에 일부 변경이 있으므로, 기본값인 `262144`에서 `sort_buffer_size` 파라미터를 업데이트해야 할 수 있습니다. 데이터베이스가 W&B 애플리케이션과 효율적으로 작동하도록 값이 `33554432(32MiB)`로 설정되는 것이 좋습니다. 이는 MySQL 버전 8.0.28 이상에서만 작동합니다.
  
  </TabItem>
</Tabs>

### 데이터베이스 고려 사항

자체 MySQL 데이터베이스를 운영할 때 다음을 고려하세요:

1. **백업**. 데이터베이스를 별도의 시설로 주기적으로 백업해야 합니다. 최소 1주일간 보존하는 매일 백업을 제안합니다.
2. **성능.** 서버가 실행 중인 디스크는 빨라야 합니다. SSD 또는 가속화된 NAS에서 데이터베이스를 실행하는 것이 좋습니다.
3. **모니터링.** 데이터베이스는 부하에 대해 모니터링되어야 합니다. 시스템의 CPU 사용량이 5분 이상 > 40%로 지속되면 서버가 리소스가 부족한 좋은 징후일 수 있습니다.
4. **가용성.** 가용성과 내구성 요구 사항에 따라 기본 서버에서 실시간으로 모든 업데이트를 스트리밍하는 별도의 기계에서 핫 스탠바이를 구성하고 기본 서버가 충돌하거나 손상된 경우에 대비해 이를 장애 조치(failover)로 사용할 수 있습니다.

다음 SQL 쿼리를 사용하여 데이터베이스와 사용자를 생성하세요. `SOME_PASSWORD`를 선택한 비밀번호로 교체하세요:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### 파라미터 그룹 구성

데이터베이스 성능을 튜닝하기 위해 다음 파라미터 그룹이 설정되었는지 확인하세요:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 33554432
```

## 오브젝트 스토어
오브젝트 스토어는 [Minio 클러스터](https://docs.min.io/minio/k8s/)에서 외부로 호스팅되거나 서명된 URL을 지원하는 Amazon S3 호환 오브젝트 스토어에서 실행할 수 있습니다. 오브젝트 스토어가 서명된 URL을 지원하는지 확인하려면 [다음 스크립트](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b)를 실행하세요.

또한, 다음 CORS 정책을 오브젝트 스토어에 적용해야 합니다.

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

Amazon S3 호환 오브젝트 스토어에 연결할 때 자격 증명을 연결 문자열에 지정할 수 있습니다. 예를 들어, 다음을 지정할 수 있습니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

오브젝트 스토어에 대해 신뢰할 수 있는 SSL 인증서를 구성한 경우 W&B가 TLS를 통해서만 연결하도록 지시할 수 있습니다. 이를 위해 URL에 `tls` 쿼리 매개변수를 추가하세요. 예를 들어, 다음 URL 예시는 Amazon S3 URI에 TLS 쿼리 매개변수를 추가하는 방법을 보여줍니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::caution
이것은 SSL 인증서가 신뢰할 수 있을 때만 작동합니다. W&B는 자체 서명된 인증서를 지원하지 않습니다.
:::

`BUCKET_QUEUE`를 `internal://`로 설정하면 제3자 오브젝트 스토어를 사용합니다. 이는 W&B 서버가 외부 SQS 큐 또는 동등한 것에 의존하는 대신 모든 오브젝트 알림을 내부적으로 관리하도록 지시합니다.

자체 오브젝트 스토어를 운영할 때 고려해야 할 가장 중요한 사항은 다음과 같습니다:

1. **저장 용량 및 성능**. 자기 디스크를 사용하는 것은 괜찮지만, 이러한 디스크의 용량을 모니터링해야 합니다. 평균 W&B 사용량은 수십에서 수백 기가바이트의 결과를 초래합니다. 과도한 사용은 페타바이트의 스토리지 소비로 이어질 수 있습니다.
2. **고장 허용**. 최소한, 오브젝트를 저장하는 물리적 디스크는 RAID 어레이에 있어야 합니다. minio를 사용하는 경우, [분산 모드](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed)에서 실행하는 것을 고려하세요.
3. **가용성.** 스토리지가 사용 가능한지 확인하기 위해 모니터링을 구성해야 합니다.

자체 오브젝트 스토리지 서비스를 운영하는 대안으로 많은 엔터프라이즈 옵션이 있습니다:

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO 설정

minio를 사용하는 경우, 버킷을 생성하기 위해 다음 명령을 실행할 수 있습니다.

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## Kubernetes 배포

다음 k8s yaml은 사용자 정의할 수 있지만, Kubernetes에서 로컬을 구성하기 위한 기본 기초로 사용될 수 있습니다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name