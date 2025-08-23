---
title: W&B 멀티 테넌트 클라우드 사용하기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud는 W&B의 Google Cloud Platform (GCP) 계정 내에 배포되는 완전 관리형 플랫폼입니다. [GCP의 북미 리전](https://cloud.google.com/compute/docs/regions-zones)에서 운영됩니다. W&B Multi-tenant Cloud는 GCP의 오토스케일링 기능을 활용하여, 트래픽 증감에 따라 플랫폼이 적절하게 확장 또는 축소되도록 보장합니다.

{{< img src="/images/hosting/saas_cloud_arch.png" alt="Multi-tenant Cloud architecture diagram" >}}

W&B Multi-tenant Cloud는 조직의 요구에 맞춰 확장되며, 프로젝트당 최대 250,000개의 메트릭과 각 메트릭당 최대 100만 개의 데이터 포인트 로깅을 지원합니다. 더 큰 규모의 배포가 필요한 경우 [support](mailto:support@wandb.com)로 문의해 주세요.

## 데이터 보안

Free 또는 Pro 플랜 사용자의 경우, 모든 데이터는 공유 클라우드 스토리지에만 저장되며 공유 클라우드 컴퓨트 서비스로 처리됩니다. 요금제에 따라 저장 용량 제한이 있을 수 있습니다.

Enterprise 플랜 사용자는 [보안 스토리지 커넥터를 활용한 BYOB(Bring Your Own Bucket)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 통해 [팀 단위로]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ko" >}}) 모델, 데이터셋 등 다양한 파일을 저장할 수 있습니다. 여러 팀에서 단일 버킷을 사용하도록 설정할 수도 있고, 각 W&B Teams별로 버킷을 분리해 사용할 수도 있습니다. 팀에 BYOB를 설정하지 않을 경우 해당 팀의 데이터는 공유 클라우드 스토리지에 저장됩니다.

배포 환경이 귀하 조직의 정책과 [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide)를 준수하도록 책임은 사용자에게 있습니다.

## 아이덴티티 및 엑세스 관리 (IAM)
Enterprise 플랜에서는 추가적인 아이덴티티 및 엑세스 관리 기능을 통해 W&B 배포 환경에 대한 안전한 인증 및 효과적인 권한 부여가 가능합니다.

* OIDC 또는 SAML을 이용한 SSO 인증 지원. 조직에 SSO 구성이 필요하다면 W&B 팀이나 support에 문의해 주세요.
* 조직 범위와 팀 내에서 [적절한 사용자 역할을 설정]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ko" >}})할 수 있습니다.
* [Restricted projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}}) 기능을 이용하여 W&B 프로젝트의 범위를 정의하고, 누가 해당 프로젝트를 조회, 수정, 그리고 W&B run을 제출할 수 있을지 제한할 수 있습니다.

## 모니터링
조직 관리자는 계정 화면 내 `Billing` 탭에서 사용량과 결제 내역을 관리할 수 있습니다. Multi-tenant Cloud의 공유 클라우드 스토리지를 사용하는 경우, 관리자는 조직 내 다양한 팀에서 스토리지 사용량을 최적화할 수 있습니다.

## 유지 관리
W&B Multi-tenant Cloud는 멀티테넌트 기반의 완전 관리형 플랫폼입니다. W&B에서 직접 관리하므로, 사용자는 W&B 플랫폼을 직접 구축·운영하는 오버헤드나 비용이 발생하지 않습니다.

## 컴플라이언스  
Multi-tenant Cloud의 보안 통제는 주기적으로 내부 및 외부 감사를 받습니다. SOC2 리포트 및 기타 보안·컴플라이언스 관련 자료를 요청하려면 [W&B Security Portal](https://security.wandb.ai/)을 참고하세요.

## 다음 단계
[Multi-tenant Cloud에 직접 엑세스](https://wandb.ai)하여 대부분의 기능을 무료로 시작할 수 있습니다. 데이터 보안 및 IAM 고급 기능을 체험하려면 [Enterprise 트라이얼 신청](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial)을 해보세요.