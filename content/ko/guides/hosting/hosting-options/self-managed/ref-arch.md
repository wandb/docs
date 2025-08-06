---
title: 참조 아키텍처
description: W&B 참조 아키텍처
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-ref-arch
    parent: self-managed
weight: 1
---

이 페이지에서는 W&B 배포를 위한 참조 아키텍처를 설명하고, 프로덕션 환경에서 플랫폼을 운영하기 위해 권장되는 인프라와 리소스를 안내합니다.

W&B의 배포 환경에 따라 다양한 서비스가 배포의 탄력성을 높이기 위해 활용될 수 있습니다.

예를 들어, 주요 클라우드 제공업체는 데이터베이스 설정, 유지 관리, 고가용성, 그리고 복원력을 단순화할 수 있는 강력한 매니지드 데이터베이스 서비스를 제공합니다.

이 참조 아키텍처는 일부 일반적인 배포 시나리오를 다루며, W&B 배포를 클라우드 벤더 서비스와 통합하여 최적의 성능과 안정성을 확보하는 방법을 보여줍니다.

## 시작하기 전에

어떤 애플리케이션이든 프로덕션 환경에서 운영할 때 고유의 과제가 있으며, W&B도 예외는 아닙니다. 우리는 전체 과정을 간소화하는 것을 목표로 하지만, 각자의 아키텍처와 설계 결정에 따라 특정 복잡성이 발생할 수 있습니다. 일반적으로 프로덕션 배포를 관리한다는 것은 하드웨어, 운영 체제, 네트워킹, 스토리지, 보안, W&B 플랫폼 자체 및 기타 의존성을 포함한 다양한 구성요소를 관리하는 것을 의미합니다. 이 책임은 환경의 초기 구축은 물론, 지속적인 유지 관리까지 포함합니다.

W&B의 직접 관리형 셀프 서비스 방식이 팀과 요구사항에 적합한지 신중히 검토하세요.

프로덕션급 애플리케이션을 운영·유지하는 데 대한 충분한 이해가 선행되어야 셀프 관리형 W&B를 배포할 수 있습니다. 팀에서 도움이 필요하다면, 당사의 Professional Services 팀 및 파트너의 구현 및 최적화 지원을 받을 수 있습니다.

직접 관리하지 않고 W&B를 운영하는 매니지드 솔루션에 대해 더 알아보고 싶다면 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})와 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 참고하세요.

## 인프라

{{< img src="/images/hosting/reference_architecture.png" alt="W&B 인프라 다이어그램" >}}

### 애플리케이션 계층

애플리케이션 계층은 다중 노드 Kubernetes 클러스터로 구성되어, 노드 장애에도 복구할 수 있도록 설계되어 있습니다. Kubernetes 클러스터는 W&B의 pod를 실행하고 관리합니다.

### 스토리지 계층

스토리지 계층은 MySQL 데이터베이스와 오브젝트 스토리지로 구성되어 있습니다. MySQL 데이터베이스는 메타데이터를 저장하고, 오브젝트 스토리지는 모델과 데이터셋 등의 Artifacts를 저장합니다.

## 인프라 요구사항

### Kubernetes
W&B Server 애플리케이션은 여러 pod를 배포하는 [Kubernetes Operator]({{< relref path="kubernetes-operator/" lang="ko" >}})로 배포됩니다. 따라서, W&B는 다음을 갖춘 Kubernetes 클러스터를 필요로 합니다:
- 완전히 구성되어 작동하는 Ingress 컨트롤러
- Persistent Volume 프로비저닝 기능

### MySQL
W&B는 메타데이터를 MySQL 데이터베이스에 저장합니다. 데이터베이스의 성능·스토리지 요구사항은 모델 파라미터 및 관련 메타데이터의 구조에 따라 달라집니다. 예를 들어 트레이닝 run을 더 많이 추적할수록 데이터베이스 크기는 증가하며, run 테이블, 사용자 워크스페이스, 리포트에 대한 쿼리에 따라 데이터베이스 로드도 높아집니다.

셀프 관리형 MySQL 데이터베이스를 배포할 때는 다음을 고려하세요.

- **백업**: 별도의 설비에 정기적으로 데이터베이스를 백업해야 합니다. W&B는 최소 1주일 간의 보존 기간으로 일일 백업을 권장합니다.
- **성능**: 서버가 설치된 디스크는 빠른 것이 좋습니다. W&B는 SSD 또는 가속 NAS에서 데이터베이스 구동을 권장합니다.
- **모니터링**: 데이터베이스의 로드는 반드시 모니터링해야 합니다. CPU 사용률이 시스템의 40%를 넘는 상태가 5분 이상 지속되면 리소스가 부족하다는 신호일 수 있습니다.
- **가용성**: 가용성과 내구성 요구사항에 따라, 메인 서버에서 실시간으로 모든 업데이트를 스트리밍하여 장애 시 즉시 페일오버할 수 있는 별도의 머신에 핫 스탠바이를 구성할 수 있습니다.

### 오브젝트 스토리지
W&B는 pre-signed URL과 CORS를 지원하는 오브젝트 스토리지를 필요로 하며, 다음 중 하나로 배포할 수 있습니다:

- [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage): AI 워크로드에 최적화된 고성능, S3 호환 오브젝트 스토리지 서비스입니다.
- [Amazon S3](https://aws.amazon.com/s3/): 업계 최고의 확장성, 데이터 가용성, 보안, 성능을 제공하는 오브젝트 스토리지 서비스입니다.
- [Google Cloud Storage](https://cloud.google.com/storage): 대규모 비정형 데이터를 저장하기 위한 매니지드 서비스입니다.
- [Azure Blob Storage](https://azure.microsoft.com/products/storage/blobs): 텍스트, 이진 데이터, 이미지, 동영상, 로그 등 대규모 비정형 데이터를 저장하는 클라우드 기반 오브젝트 스토리지 솔루션입니다.
- S3 호환 스토리지(예: [MinIO](https://github.com/minio/minio))를 직접 클라우드 또는 온프레미스 인프라에 호스팅할 수 있습니다.

### 버전
| 소프트웨어    | 최소 버전                                  |
| ------------ | -------------------------------------------- |
| Kubernetes   | v1.29                                        |
| MySQL        | v8.0.0, "General Availability" 릴리즈 한정   |

### 네트워킹

네트워크 배포의 경우, 설치와 런타임 _모두_ 다음 엔드포인트로의 egress가 필요합니다:
* https://deploy.wandb.ai
* https://charts.wandb.ai
* https://docker.io
* https://quay.io
* `https://gcr.io`

에어갭 환경 배포에 대해서는 [Kubernetes operator for air-gapped instances]({{< relref path="kubernetes-operator/operator-airgapped.md" lang="ko" >}})를 참고하세요.
트레이닝 인프라와 실험 추적을 위한 각 시스템에서는 W&B 및 오브젝트 스토리지에 엑세스할 수 있어야 합니다.

### DNS
W&B 배포의 완전한 도메인 이름(FQDN)은 Ingress/로드 밸런서의 IP 어드레스를 가리키는 A 레코드로 등록되어야 합니다.

### SSL/TLS
W&B는 클라이언트와 서버 간 보안 통신을 위해 유효하게 서명된 SSL/TLS 인증서를 요구합니다. SSL/TLS 종료는 반드시 Ingress/로드 밸런서에서 처리되어야 하며, W&B Server 애플리케이션에서는 SSL 또는 TLS 연결을 직접 종료하지 않습니다.

참고: W&B는 셀프 서명 인증서 또는 커스텀 CA 사용을 권장하지 않습니다.

### 지원 CPU 아키텍처
W&B는 Intel(x86) CPU 아키텍처에서 동작합니다. ARM은 지원하지 않습니다.

## 인프라 프로비저닝
프로덕션용 W&B 배포에는 Terraform 사용을 권장합니다. Terraform을 사용하면 필요한 리소스, 다른 리소스와의 참조 관계, 의존성을 정의할 수 있습니다. W&B는 주요 클라우드 제공업체용 Terraform 모듈을 제공합니다. 자세한 내용은 [Deploy W&B Server within self managed cloud accounts]({{< relref path="/guides/hosting/hosting-options/self-managed.md#deploy-wb-server-within-self-managed-cloud-accounts" lang="ko" >}})를 참고하세요.

## 사이징
배포를 계획할 때 다음의 일반적인 가이드라인을 출발점으로 삼으세요. W&B는 신규 배포 환경의 모든 구성 요소를 면밀히 모니터링하고, 실제 사용 패턴을 바탕으로 조정할 것을 권장합니다. 프로덕션 배포도 시간이 지남에 따라 계속 모니터링하며, 최적의 성능 유지를 위해 필요에 따라 조정하십시오.

### Models만 운영 시

#### Kubernetes

| 환경          | CPU	        | 메모리	    | 디스크              | 
| ------------- | --------------- | -------------- | ------------------- | 
| Test/Dev      | 2 cores         | 16 GB          | 100 GB              |
| Production    | 8 cores         | 64 GB          | 100 GB              |

Kubernetes 워커 노드당 기준입니다.

#### MySQL

| 환경          | CPU	        | 메모리	    | 디스크              | 
| ------------- | --------------- | -------------- | ------------------- | 
| Test/Dev      | 2 cores         | 16 GB          | 100 GB              |
| Production    | 8 cores         | 64 GB          | 500 GB              |

MySQL 노드당 기준입니다.

### Weave만 운영 시

#### Kubernetes

| 환경          | CPU            | 메모리         | 디스크              | 
| ------------- | --------------- | -------------- | ------------------- | 
| Test/Dev      | 4 cores         | 32 GB          | 100 GB              |
| Production    | 12 cores        | 96 GB          | 100 GB              |

Kubernetes 워커 노드당 기준입니다.

#### MySQL

| 환경          | CPU            | 메모리         | 디스크              | 
| ------------- | --------------- | -------------- | ------------------- | 
| Test/Dev      | 2 cores         | 16 GB          | 100 GB              |
| Production    | 8 cores         | 64 GB          | 500 GB              |

MySQL 노드당 기준입니다.

### Models와 Weave 모두 운영 시

#### Kubernetes

| 환경          | CPU            | 메모리         | 디스크              | 
| ------------- | --------------- | -------------- | ------------------- | 
| Test/Dev      | 4 cores         | 32 GB          | 100 GB              |
| Production    | 16 cores        | 128 GB         | 100 GB              |

Kubernetes 워커 노드당 기준입니다.

#### MySQL

| 환경          | CPU            | 메모리         | 디스크              | 
| ------------- | --------------- | -------------- | ------------------- | 
| Test/Dev      | 2 cores         | 16 GB          | 100 GB              |
| Production    | 8 cores         | 64 GB          | 500 GB              |

MySQL 노드당 기준입니다.

## 클라우드 제공업체 인스턴스 권장사항

### 서비스

| 클라우드   | Kubernetes	 | MySQL	                | 오브젝트 스토리지              |   
| ---------- | ------------ | -------------------- | ------------------------------ | 
| AWS        | EKS          | RDS Aurora           | S3                             |
| GCP        | GKE          | Google Cloud SQL - Mysql | Google Cloud Storage (GCS) |
| Azure      | AKS          | Azure Database for Mysql | Azure Blob Storage         |

### 머신 타입

이 권장 사항은 셀프 관리형 W&B 배포의 각 노드에 적용됩니다.

#### AWS

| 환경       | K8s (Models만)  | K8s (Weave만)   | K8s (Models&Weave)  | MySQL	            |  
| ---------- | ---------------- | --------------- | --------------------| ------------------ |  
| Test/Dev   | r6i.large        | r6i.xlarge      | r6i.xlarge          | db.r6g.large       | 
| Production | r6i.2xlarge      | r6i.4xlarge     | r6i.4xlarge         | db.r6g.2xlarge     | 

#### GCP

| 환경       | K8s (Models만)  | K8s (Weave만)   | K8s (Models&Weave)  | MySQL             |  
| ---------- | ---------------- | --------------- | --------------------| ----------------- |  
| Test/Dev   | n2-highmem-2     | n2-highmem-4    | n2-highmem-4        | db-n1-highmem-2   | 
| Production | n2-highmem-8     | n2-highmem-16   | n2-highmem-16       | db-n1-highmem-8   | 

#### Azure

| 환경       | K8s (Models만)    | K8s (Weave만)     | K8s (Models&Weave)   | MySQL                |  
| ---------- | ------------------ | ----------------- | -------------------- | ------------------- |  
| Test/Dev   | Standard_E2_v5     | Standard_E4_v5    | Standard_E4_v5       | MO_Standard_E2ds_v4 | 
| Production | Standard_E8_v5     | Standard_E16_v5   | Standard_E16_v5      | MO_Standard_E8ds_v4 |
