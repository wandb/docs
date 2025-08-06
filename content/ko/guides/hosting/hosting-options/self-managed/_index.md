---
title: 셀프 매니지드
description: 프로덕션 환경에서 W&B 배포하기
cascade:
- url: guides/hosting/self-managed/:filename
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: guides/hosting/hosting-options/self-managed
---

## W&B 셀프 매니지드 클라우드 또는 온프레미스 인프라에서 사용하기

{{% alert %}}
W&B는 [W&B 멀티 테넌트 클라우드]({{< relref path="../saas_cloud.md" lang="ko" >}}) 또는 [W&B 전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})와 같은 완전 관리형 배포 옵션을 권장합니다. W&B 완전 관리형 서비스는 최소한의 설정만으로도 간편하고 안전하게 이용할 수 있습니다.
{{% /alert %}}

[AWS, GCP, Azure 클라우드 계정]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ko" >}}) 또는 [온프레미스 인프라]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ko" >}}) 내에 W&B Server를 배포할 수 있습니다.

귀하의 IT/DevOps/MLOps 팀은 다음 업무를 담당합니다:
- 배포 환경 프로비저닝
- 조직의 정책 및 [보안 기술 구현 가이드 (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) (적용 가능한 경우)에 따른 인프라 보안 관리
- 업그레이드 및 패치 적용
- 셀프 매니지드 W&B Server 인스턴스의 지속적인 운영 및 유지 관리





## 셀프 매니지드 클라우드 계정 내에 W&B Server 배포

AWS, GCP, Azure 클라우드 계정 내에 W&B Server를 배포하려면, 공식 W&B Terraform 스크립트 사용을 권장합니다.

[AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ko" >}}), [GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ko" >}}), [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ko" >}})에서 W&B Server를 설정하는 방법에 대한 자세한 내용은 각 클라우드별 문서를 참고하세요.

## 온프레미스 인프라에서 W&B Server 배포

온프레미스 인프라에 W&B Server를 설치하려면 여러 인프라 구성 요소를 설정해야 합니다. 필요한 주요 구성 요소는 다음과 같습니다 (단, 이에 한정되지는 않습니다):

- (강력 추천) Kubernetes 클러스터
- MySQL 8 데이터베이스 클러스터
- Amazon S3 호환 오브젝트 스토리지
- Redis 캐시 클러스터

온프레미스 인프라에서 W&B Server를 설치하는 방법에 대한 자세한 내용은 [온프레미스 인프라에 설치하기]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ko" >}})를 참고하세요. W&B는 각 구성 요소에 대한 권장 사항과 설치 과정 전반에 대한 가이드를 제공할 수 있습니다.

## 맞춤형 클라우드 플랫폼에서 W&B Server 배포

AWS, GCP, Azure가 아닌 클라우드 플랫폼에도 W&B Server를 배포할 수 있습니다. 이 경우 요구 사항은 [온프레미스 인프라]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ko" >}})에서 배포하는 것과 유사합니다.

## W&B Server 라이선스 획득하기

W&B server의 설정을 완료하려면 W&B 트라이얼 라이선스가 필요합니다. [Deploy Manager](https://deploy.wandb.ai/deploy)에서 무료 트라이얼 라이선스를 발급받을 수 있습니다.

{{% alert %}}
아직 W&B 계정이 없다면, 먼저 계정을 만들어 라이선스를 발급받으세요.

보안 및 엔터프라이즈용 기능이 포함된 W&B Server용 엔터프라이즈 라이선스가 필요하다면, [이 폼을 제출](https://wandb.ai/site/for-enterprise/self-hosted-trial)하거나 W&B 담당자에게 문의해 주세요.
{{% /alert %}}

해당 URL은 **W&B Local 라이선스 받기** 폼으로 이동합니다. 다음 정보를 입력하세요:

1. **Choose Platform** 단계에서 배포 유형을 선택하세요.
2. **Basic Information** 단계에서 라이선스 소유자를 선택하거나 새 조직을 추가하세요.
3. **Get a License** 단계의 **Name of Instance** 필드에 인스턴스 이름을 입력하고 필요하다면 **Description** 필드에 설명을 입력하세요.
4. **Generate License Key** 버튼을 선택하세요.

인스턴스에 연결된 라이선스와 함께 배포 개요를 확인할 수 있는 페이지가 표시됩니다.