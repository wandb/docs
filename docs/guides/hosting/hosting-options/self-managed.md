---
title: Use self-managed cloud or on-prem infrastructure
description: W&B를 프로덕션에 배포하기
displayed_sidebar: default
---

:::info
W&B는 [W&B 멀티 테넌트 클라우드](./saas_cloud.md) 또는 [W&B 전용 클라우드](./dedicated_cloud.md) 배포 유형과 같은 완전 관리형 배포 옵션을 권장합니다. W&B 완전 관리형 서비스는 사용이 간단하고 안전하며, 최소한의 설정 혹은 설정이 필요하지 않습니다.
:::

W&B Server를 [AWS, GCP, 또는 Azure 클라우드 계정](#deploy-wb-server-within-self-managed-cloud-accounts) 또는 [온프레미스 인프라](#deploy-wb-server-in-on-prem-infrastructure) 내에 배포하세요.

귀하의 IT/DevOps/MLOps 팀은 배포를 준비하고, 업그레이드를 관리하며, 자가 관리형 W&B Server 인스턴스를 지속적으로 유지 관리할 책임이 있습니다.

## 자가 관리형 클라우드 계정 내에 W&B Server 배포

W&B는 공식 W&B Terraform 스크립트를 사용하여 AWS, GCP 또는 Azure 클라우드 계정에 W&B Server를 배포하는 것을 권장합니다.

추가적인 정보는 [AWS](../self-managed/aws-tf.md), [GCP](../self-managed/gcp-tf.md) 또는 [Azure](../self-managed/azure-tf.md)에서 W&B Server를 설정하는 방법 관련 정보를 참조하세요.

## 온프레미스 인프라에 W&B Server 배포

온프레미스 인프라에 W&B Server를 설정하려면 여러 인프라 구성 요소를 설정해야 합니다. 그 구성 요소에는 다음이 포함되지만 이에 국한되지 않습니다:

- (강력 추천) Kubernetes 클러스터
- MySQL 8 데이터베이스 클러스터
- Amazon S3 호환 오브젝트 스토리지
- Redis 캐시 클러스터

온프레미스 인프라에 W&B Server를 설치하는 방법에 대한 자세한 내용은 [온프레미스 인프라에 설치하기](../self-managed/bare-metal.md)를 참조하세요. W&B는 다양한 구성 요소에 대한 권장 사항을 제공하고 설치 프로세스를 안내할 수 있습니다.

## 커스텀 클라우드 플랫폼에 W&B Server 배포

AWS, GCP 또는 Azure가 아닌 클라우드 플랫폼에 W&B Server를 배포할 수 있습니다. 이를 위한 요구 사항은 [온프레미스 인프라](#deploy-wb-server-in-on-prem-infrastructure)에 배포하는 것과 유사합니다.

## W&B Server 라이센스 획득

W&B server 설정을 완료하기 위해서는 W&B 트라이얼 라이센스가 필요합니다. [Deploy Manager](https://deploy.wandb.ai/deploy)를 열어 무료 트라이얼 라이센스를 생성하세요.

:::note
만약 W&B 계정이 없다면 무료 라이센스를 생성하기 위해 계정을 새로 만들어야 합니다.
:::

URL은 **W&B Local 라이센스 획득** 폼으로 리디렉션됩니다. 다음 정보를 제공하세요:

1. **Choose Platform** 단계에서 배포 유형을 선택합니다.
2. **기본 정보** 단계에서 라이센스 소유자를 선택하거나 새 조직을 추가합니다.
3. **Get a License** 단계에서 인스턴스의 이름을 **Name of Instance** 필드에 입력하고, 선택적으로 **Description** 필드에 설명을 제공합니다.
4. **Generate License Key** 버튼을 선택합니다.

인스턴스에 연결된 라이센스와 함께 배포 개요 페이지가 표시됩니다.

:::info
중요한 보안 및 기타 엔터프라이즈 친화적인 기능을 포함하는 W&B Server의 엔터프라이즈 라이센스가 필요하다면 [이 폼을 제출하세요](https://wandb.ai/site/for-enterprise/self-hosted-trial) 또는 귀하의 W&B 팀에 문의하세요.
:::