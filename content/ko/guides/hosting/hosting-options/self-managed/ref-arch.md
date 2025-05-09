---
title: Reference Architecture
description: W&B 참조 아키텍처
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-ref-arch
    parent: self-managed
weight: 1
---

이 페이지에서는 Weights & Biases 배포를 위한 참조 아키텍처를 설명하고 플랫폼의 프로덕션 배포를 지원하기 위해 권장되는 인프라 및 리소스를 간략하게 설명합니다.

Weights & Biases(W&B)에 대해 선택한 배포 환경에 따라 다양한 서비스를 통해 배포의 복원력을 향상시킬 수 있습니다.

예를 들어 주요 클라우드 공급자는 데이터베이스 구성, 유지 관리, 고가용성 및 복원력의 복잡성을 줄이는 데 도움이 되는 강력한 관리형 데이터베이스 서비스를 제공합니다.

이 참조 아키텍처는 몇 가지 일반적인 배포 시나리오를 다루고 최적의 성능과 안정성을 위해 W&B 배포를 클라우드 공급업체 서비스와 통합하는 방법을 보여줍니다.

## 시작하기 전에

모든 애플리케이션을 프로덕션 환경에서 실행하는 데에는 자체적인 어려움이 따르며 W&B도 예외는 아닙니다. Google은 프로세스를 간소화하는 것을 목표로 하지만 고유한 아키텍처 및 설계 결정에 따라 특정 복잡성이 발생할 수 있습니다. 일반적으로 프로덕션 배포를 관리하려면 하드웨어, 운영 체제, 네트워킹, 스토리지, 보안, W&B 플랫폼 자체 및 기타 종속성을 포함한 다양한 구성 요소를 감독해야 합니다. 이 책임은 환경의 초기 설정과 지속적인 유지 관리 모두에 적용됩니다.

W&B를 사용한 자체 관리 방식이 팀 및 특정 요구 사항에 적합한지 신중하게 고려하십시오.

프로덕션 등급 애플리케이션을 실행하고 유지 관리하는 방법에 대한 확실한 이해는 자체 관리 W&B를 배포하기 전에 중요한 전제 조건입니다. 팀에 지원이 필요한 경우 Google의 Professional Services 팀과 파트너가 구현 및 최적화에 대한 지원을 제공합니다.

직접 관리하는 대신 W&B 실행을 위한 관리형 솔루션에 대한 자세한 내용은 [W&B 멀티 테넌트 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 및 [W&B 전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 참조하십시오.

## 인프라

{{< img src="/images/hosting/reference_architecture.png" alt="W&B 인프라 다이어그램" >}}

### 애플리케이션 레이어

애플리케이션 레이어는 노드 장애에 대한 복원력을 갖춘 다중 노드 Kubernetes 클러스터로 구성됩니다. Kubernetes 클러스터는 W&B의 포드를 실행하고 유지 관리합니다.

### 스토리지 레이어

스토리지 레이어는 MySQL 데이터베이스와 오브젝트 스토리지로 구성됩니다. MySQL 데이터베이스는 메타데이터를 저장하고 오브젝트 스토리지는 모델 및 데이터셋과 같은 아티팩트를 저장합니다.

## 인프라 요구 사항

### Kubernetes
W&B Server 애플리케이션은 여러 포드를 배포하는 [Kubernetes Operator]({{< relref path="kubernetes-operator/" lang="ko" >}})로 배포됩니다. 이러한 이유로 W&B에는 다음이 포함된 Kubernetes 클러스터가 필요합니다.
- 완전히 구성되고 작동하는 Ingress 컨트롤러.
- Persistent Volumes를 프로비저닝하는 기능.

### MySQL
W&B는 메타데이터를 MySQL 데이터베이스에 저장합니다. 데이터베이스의 성능 및 스토리지 요구 사항은 모델 파라미터 및 관련 메타데이터의 형태에 따라 달라집니다. 예를 들어 더 많은 트레이닝 run을 추적할수록 데이터베이스 크기가 커지고 run 테이블, 사용자 워크스페이스 및 리포트의 쿼리를 기반으로 데이터베이스에 대한 부하가 증가합니다.

자체 관리 MySQL 데이터베이스를 배포할 때 다음 사항을 고려하십시오.

- **백업**. 데이터베이스를 별도의 시설에 주기적으로 백업해야 합니다. W&B는 최소 1주일의 보존 기간으로 매일 백업하는 것이 좋습니다.
- **성능**. 서버가 실행되는 디스크는 빨라야 합니다. W&B는 SSD 또는 가속화된 NAS에서 데이터베이스를 실행하는 것이 좋습니다.
- **모니터링**. 데이터베이스의 부하를 모니터링해야 합니다. CPU 사용량이 시스템의 40%를 초과하여 5분 이상 유지되면 서버에 리소스가 부족하다는 좋은 징조일 수 있습니다.
- **가용성**. 가용성 및 내구성 요구 사항에 따라 기본 서버에서 모든 업데이트를 실시간으로 스트리밍하고 기본 서버가 충돌하거나 손상될 경우 장애 조치하는 데 사용할 수 있는 별도의 시스템에서 핫 스탠바이를 구성할 수 있습니다.

### 오브젝트 스토리지
W&B에는 사전 서명된 URL 및 CORS 지원이 포함된 오브젝트 스토리지가 필요하며 다음 중 하나로 배포됩니다.
- Amazon S3
- Azure Cloud Storage
- Google Cloud Storage
- Amazon S3와 호환되는 스토리지 서비스

### 버전
| 소프트웨어     | 최소 버전                              |
| ------------ | -------------------------------------------- |
| Kubernetes   | v1.29                                        |
| MySQL        | v8.0.0, "General Availability" 릴리스만 해당 |

### 네트워킹

네트워크로 연결된 배포의 경우 설치 및 런타임 _모두_ 중에 이러한 엔드포인트에 대한 송신이 필요합니다.
* https://deploy.wandb.ai
* https://charts.wandb.ai
* https://docker.io
* https://quay.io
* `https://gcr.io`

에어 갭 배포에 대한 자세한 내용은 [에어 갭 인스턴스용 Kubernetes operator]({{< relref path="kubernetes-operator/operator-airgapped.md" lang="ko" >}})를 참조하십시오.
W&B와 오브젝트 스토리지에 대한 엑세스는 트레이닝 인프라와 Experiments의 요구 사항을 추적하는 각 시스템에 필요합니다.

### DNS
W&B 배포의 정규화된 도메인 이름(FQDN)은 A 레코드를 사용하여 수신/로드 밸런서의 IP 어드레스로 확인되어야 합니다.

### SSL/TLS
W&B에는 클라이언트와 서버 간의 보안 통신을 위한 유효한 서명된 SSL/TLS 인증서가 필요합니다. SSL/TLS 종료는 수신/로드 밸런서에서 발생해야 합니다. W&B Server 애플리케이션은 SSL 또는 TLS 연결을 종료하지 않습니다.

참고: W&B는 자체 서명된 인증서 및 사용자 지정 CA의 사용을 권장하지 않습니다.

### 지원되는 CPU 아키텍처
W&B는 Intel(x86) CPU 아키텍처에서 실행됩니다. ARM은 지원되지 않습니다.

## 인프라 프로비저닝
Terraform은 프로덕션 환경을 위해 W&B를 배포하는 데 권장되는 방법입니다. Terraform을 사용하면 필요한 리소스, 다른 리소스에 대한 참조 및 종속성을 정의합니다. W&B는 주요 클라우드 공급자를 위한 Terraform 모듈을 제공합니다. 자세한 내용은 [자체 관리 클라우드 계정 내에서 W&B Server 배포]({{< relref path="/guides/hosting/hosting-options/self-managed.md#deploy-wb-server-within-self-managed-cloud-accounts" lang="ko" >}})를 참조하십시오.

## 크기 조정
배포를 계획할 때 다음 일반 지침을 시작점으로 사용하십시오. W&B는 새로운 배포의 모든 구성 요소를 면밀히 모니터링하고 관찰된 사용 패턴에 따라 조정하는 것이 좋습니다. 시간이 지남에 따라 프로덕션 배포를 계속 모니터링하고 최적의 성능을 유지하기 위해 필요에 따라 조정하십시오.

### 모델만 해당

#### Kubernetes

| 환경      | CPU	            | 메모리	         | 디스크               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| 테스트/개발         | 2 코어            | 16GB              | 100GB             |
| 프로덕션       | 8 코어            | 64GB              | 100GB             |

숫자는 Kubernetes 작업자 노드당 개수입니다.

#### MySQL

| 환경      | CPU	            | 메모리	         | 디스크               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| 테스트/개발         | 2 코어            | 16GB              | 100GB             |
| 프로덕션       | 8 코어            | 64GB              | 500GB             |

숫자는 MySQL 노드당 개수입니다.


### Weave만 해당

#### Kubernetes

| 환경      | CPU                | 메모리             | 디스크               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| 테스트/개발         | 4 코어            | 32GB              | 100GB             |
| 프로덕션       | 12 코어           | 96GB              | 100GB             |

숫자는 Kubernetes 작업자 노드당 개수입니다.

#### MySQL

| 환경      | CPU                | 메모리             | 디스크               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| 테스트/개발         | 2 코어            | 16GB              | 100GB             |
| 프로덕션       | 8 코어            | 64GB              | 500GB             |

숫자는 MySQL 노드당 개수입니다.

### 모델 및 Weave

#### Kubernetes

| 환경      | CPU                | 메모리             | 디스크               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| 테스트/개발         | 4 코어            | 32GB              | 100GB             |
| 프로덕션       | 16 코어          | 128GB             | 100GB             |

숫자는 Kubernetes 작업자 노드당 개수입니다.

#### MySQL

| 환경      | CPU                | 메모리             | 디스크               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| 테스트/개발         | 2 코어            | 16GB              | 100GB             |
| 프로덕션       | 8 코어            | 64GB              | 500GB             |

숫자는 MySQL 노드당 개수입니다.

## 클라우드 공급자 인스턴스 권장 사항

### 서비스

| 클라우드       | Kubernetes	 | MySQL	                | 오브젝트 스토리지             |   
| ----------- | ------------ | ------------------------ | -------------------------- | 
| AWS         | EKS          | RDS Aurora               | S3                         |
| GCP         | GKE          | Google Cloud SQL - Mysql | Google Cloud Storage (GCS) |
| Azure       | AKS          | Azure Database for Mysql | Azure Blob Storage         |


### 머신 유형

이러한 권장 사항은 클라우드 인프라에서 W&B의 자체 관리 배포의 각 노드에 적용됩니다.

#### AWS

| 환경 | K8s (모델만 해당)  | K8s (Weave만 해당)   | K8s (모델&Weave)  | MySQL	           |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------ |  
| 테스트/개발    | r6i.large          | r6i.xlarge         | r6i.xlarge          | db.r6g.large       | 
| 프로덕션  | r6i.2xlarge        | r6i.4xlarge        | r6i.4xlarge         | db.r6g.2xlarge     | 

#### GCP

| 환경 | K8s (모델만 해당)  | K8s (Weave만 해당)   | K8s (모델&Weave)  | MySQL              |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------ |  
| 테스트/개발    | n2-highmem-2       | n2-highmem-4       | n2-highmem-4        | db-n1-highmem-2    | 
| 프로덕션  | n2-highmem-8       | n2-highmem-16      | n2-highmem-16       | db-n1-highmem-8    | 

#### Azure

| 환경 | K8s (모델만 해당)  | K8s (Weave만 해당)   | K8s (모델&Weave)  | MySQL               |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------- |  
| 테스트/개발    | Standard_E2_v5     | Standard_E4_v5     | Standard_E4_v5      | MO_Standard_E2ds_v4 | 
| 프로덕션  | Standard_E8_v5     | Standard_E16_v5    | Standard_E16_v5     | MO_Standard_E8ds_v4 |
