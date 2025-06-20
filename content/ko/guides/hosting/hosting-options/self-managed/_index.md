---
title: Self-managed
description: 프로덕션 환경에 W&B 배포
cascade:
- url: /ko/guides//hosting/self-managed/:filename
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: /ko/guides//hosting/hosting-options/self-managed
---

## 자체 관리형 클라우드 또는 온-프레미스 인프라 사용

{{% alert %}}
W&B는 [W&B Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 배포 유형과 같은 완전 관리형 배포 옵션을 권장합니다. W&B 완전 관리형 서비스는 사용하기 간단하고 안전하며 최소한의 설정만 필요하거나 설정이 전혀 필요하지 않습니다.
{{% /alert %}}

[AWS, GCP 또는 Azure 클라우드 계정]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ko" >}}) 또는 [온-프레미스 인프라]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ko" >}}) 내에 W&B Server를 배포합니다.

IT/DevOps/MLOps 팀은 배포 프로비저닝, 업그레이드 관리 및 자체 관리형 W&B Server 인스턴스의 지속적인 유지 관리를 담당합니다.

## 자체 관리형 클라우드 계정 내에 W&B Server 배포

W&B는 공식 W&B Terraform 스크립트를 사용하여 AWS, GCP 또는 Azure 클라우드 계정에 W&B Server를 배포할 것을 권장합니다.

[AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ko" >}}), [GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ko" >}}) 또는 [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ko" >}})에서 W&B Server를 설정하는 방법에 대한 자세한 내용은 특정 클라우드 공급자 문서를 참조하세요.

## 온-프레미스 인프라에 W&B Server 배포

온-프레미스 인프라에서 W&B Server를 설정하려면 여러 인프라 구성 요소를 구성해야 합니다. 이러한 구성 요소에는 다음이 포함되지만 이에 국한되지는 않습니다.

- (강력 권장) Kubernetes 클러스터
- MySQL 8 데이터베이스 클러스터
- Amazon S3 호환 오브젝트 스토리지
- Redis 캐시 클러스터

온-프레미스 인프라에 W&B Server를 설치하는 방법에 대한 자세한 내용은 [온-프레미스 인프라에 설치]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ko" >}})를 참조하세요. W&B는 다양한 구성 요소에 대한 권장 사항을 제공하고 설치 프로세스에 대한 지침을 제공할 수 있습니다.

## 사용자 지정 클라우드 플랫폼에 W&B Server 배포

AWS, GCP 또는 Azure가 아닌 클라우드 플랫폼에 W&B Server를 배포할 수 있습니다. 이에 대한 요구 사항은 [온-프레미스 인프라]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ko" >}})에 배포하는 것과 유사합니다.

## W&B Server 라이선스 받기

W&B 서버 설정을 완료하려면 W&B 트라이얼 라이선스가 필요합니다. [Deploy Manager](https://deploy.wandb.ai/deploy)를 열어 무료 트라이얼 라이선스를 생성하세요.

{{% alert %}}
아직 W&B 계정이 없는 경우 계정을 만들어 무료 라이선스를 생성하세요.

중요한 보안 및 기타 엔터프라이즈 친화적인 기능을 지원하는 W&B Server용 엔터프라이즈 라이선스가 필요한 경우 [이 양식](https://wandb.ai/site/for-enterprise/self-hosted-trial)을 제출하거나 W&B 팀에 문의하세요.
{{% /alert %}}

URL은 **W&B Local용 라이선스 받기** 양식으로 리디렉션됩니다. 다음 정보를 제공하세요.

1. **플랫폼 선택** 단계에서 배포 유형을 선택합니다.
2. **기본 정보** 단계에서 라이선스 소유자를 선택하거나 새 조직을 추가합니다.
3. **라이선스 받기** 단계의 **인스턴스 이름** 필드에 인스턴스 이름을 제공하고 필요에 따라 **설명** 필드에 설명을 제공합니다.
4. **라이선스 키 생성** 버튼을 선택합니다.

인스턴스와 연결된 라이선스와 함께 배포 개요가 있는 페이지가 표시됩니다.
