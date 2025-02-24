---
title: Use W&B Multi-tenant SaaS
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud는 W&B의 Google Cloud Platform (GCP) 계정의 [GPC 북미 지역](https://cloud.google.com/compute/docs/regions-zones)에 배포된 완전 관리형 플랫폼입니다. W&B Multi-tenant Cloud는 트래픽 증가 또는 감소에 따라 플랫폼이 적절하게 확장되도록 GCP에서 자동 확장을 활용합니다.

{{< img src="/images/hosting/saas_cloud_arch.png" alt="" >}}

## 데이터 보안

Enterprise 플랜 사용자가 아닌 경우, 모든 데이터는 공유 클라우드 스토리지에만 저장되며 공유 클라우드 컴퓨팅 서비스로 처리됩니다. 가격 플랜에 따라 스토리지 제한이 적용될 수 있습니다.

Enterprise 플랜 사용자는 [보안 스토리지 커넥터를 사용하여 자체 버킷(BYOB)을 가져올 수 있습니다]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) ([팀 레벨]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ko" >}})에서 모델, 데이터셋 등과 같은 파일을 저장할 수 있습니다. 여러 Teams에 대해 단일 버킷을 구성하거나, 서로 다른 W&B Teams에 대해 별도의 버킷을 사용할 수 있습니다. 팀에 대해 보안 스토리지 커넥터를 구성하지 않으면 해당 데이터는 공유 클라우드 스토리지에 저장됩니다.

## ID 및 엑세스 관리 (IAM)
Enterprise 플랜을 사용하는 경우, W&B Organization에서 안전한 인증 및 효과적인 권한 부여를 위해 ID 및 엑세스 관리 기능을 사용할 수 있습니다. Multi-tenant Cloud의 IAM에는 다음 기능이 제공됩니다.

* OIDC 또는 SAML을 통한 SSO 인증. 조직에 대해 SSO를 구성하려면 W&B 팀 또는 지원팀에 문의하십시오.
* 조직 범위 내에서 그리고 팀 내에서 [적절한 user 역할을 구성합니다]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ko" >}}).
* W&B project의 범위를 정의하여 누가 W&B runs를 보고, 편집하고, 제출할 수 있는지 [제한된 projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})를 통해 제한합니다.

## 모니터링
Organization 관리자는 계정 보기의 `Billing` 탭에서 계정에 대한 사용량 및 청구를 관리할 수 있습니다. Multi-tenant Cloud에서 공유 클라우드 스토리지를 사용하는 경우, 관리자는 조직 내의 여러 팀에서 스토리지 사용량을 최적화할 수 있습니다.

## 유지 관리
W&B Multi-tenant Cloud는 멀티 테넌트, 완전 관리형 플랫폼입니다. W&B Multi-tenant Cloud는 W&B에서 관리하므로 W&B 플랫폼을 프로비저닝하고 유지 관리하는 데 드는 오버헤드 및 비용이 발생하지 않습니다.

## 규정 준수
Multi-tenant Cloud에 대한 보안 제어는 내부 및 외부에서 주기적으로 감사됩니다. SOC2 리포트 및 기타 보안 및 규정 준수 문서를 요청하려면 [W&B 보안 포털](https://security.wandb.ai/)을 참조하십시오.

## 다음 단계
Enterprise 기능이 필요하지 않은 경우 [Multi-tenant Cloud에 직접 엑세스하십시오](https://wandb.ai). Enterprise 플랜을 시작하려면 [이 양식](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial)을 제출하십시오.
