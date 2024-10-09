---
title: Use W&B Multi-tenant SaaS
displayed_sidebar: default
---

W&B Multi-tenant Cloud는 W&B의 Google Cloud Platform (GCP) 계정에 배포된 완전히 관리되는 플랫폼으로, [GPC의 북미 지역](https://cloud.google.com/compute/docs/regions-zones)에 위치해 있습니다. W&B Multi-tenant Cloud는 GCP의 자동 확장을 활용하여 트래픽의 증가나 감소에 따라 플랫폼이 적절히 확장되도록 보장합니다.

![](/images/hosting/saas_cloud_arch.png)

## 데이터 보안

엔터프라이즈 플랜을 사용하지 않는 사용자들의 모든 데이터는 공유 클라우드 스토리지에만 저장되며, 공유 클라우드 컴퓨팅 서비스로 처리됩니다. 요금제에 따라 스토리지 제한이 적용될 수 있습니다.

엔터프라이즈 플랜 사용자들은 [보안 스토리지 커넥터를 사용하여 자신의 버킷을 가져올 수 있습니다 (BYOB)](../data-security/secure-storage-connector.md) [팀 수준에서](../data-security/secure-storage-connector.md#configuration-options) 모델, 데이터셋 등 파일을 저장할 수 있습니다. 여러 팀에 대해 단일 버킷을 설정할 수도 있으며, 다른 W&B Teams에 대해 별도의 버킷을 사용할 수도 있습니다. 팀에 대해 보안 스토리지 커넥터를 설정하지 않으면 해당 데이터는 공유 클라우드 스토리지에 저장됩니다.

## ID 및 엑세스 관리 (IAM)

엔터프라이즈 플랜을 사용하는 경우 안전한 인증 및 효과적인 권한 부여를 위해 W&B 조직에 대한 ID 및 엑세스 관리 기능을 사용할 수 있습니다. Multi-tenant Cloud에서 IAM을 위한 다음 기능이 제공됩니다:

* OIDC 또는 SAML을 사용한 SSO 인증. 조직에 대해 SSO를 구성하려면 W&B 팀 또는 지원팀에 문의하세요.
* 조직의 범위와 팀 내에서의 [적절한 사용자 역할 구성](../iam/manage-users.md).
* [제한적 프로젝트](../iam/restricted-projects.md)를 통해 W&B 프로젝트의 범위를 정의하여 누가 W&B Runs를 볼 수 있고, 수정할 수 있으며, 제출할 수 있는지 제한합니다.

## 모니터

조직 관리자들은 계정 보기의 `Billing` 탭에서 계정의 사용량과 청구를 관리할 수 있습니다. Multi-tenant Cloud에서 공유 클라우드 스토리지를 사용하는 경우, 관리자는 조직 내 다양한 팀에 걸쳐 스토리지 사용을 최적화할 수 있습니다.

## 유지 보수

W&B Multi-tenant Cloud는 다중 테넌트, 완전 관리 플랫폼입니다. W&B Multi-tenant Cloud는 W&B에 의해 관리되므로, W&B 플랫폼을 제공하고 유지하는 오버헤드 및 비용이 발생하지 않습니다.

## 준수

Multi-tenant Cloud의 보안 제어는 주기적으로 내부 및 외부에서 감사됩니다. SOC2 보고서 및 기타 보안 및 준수 문서를 요청하려면 [W&B 보안 포털](https://security.wandb.ai/)을 참조하세요.

## 다음 단계

엔터프라이즈 기능이 필요하지 않은 경우 [Multi-tenant Cloud에 직접 엑세스](https://wandb.ai)하세요. 엔터프라이즈 플랜을 시작하려면 [이 양식](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial)을 제출하세요.