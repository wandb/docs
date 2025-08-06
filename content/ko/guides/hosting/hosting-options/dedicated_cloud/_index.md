---
title: 전용 클라우드
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## W&B Dedicated Cloud 를 통한 싱글 테넌트 SaaS 활용

W&B Dedicated Cloud 는 W&B 가 AWS, GCP, 또는 Azure 클라우드 계정에 배포하는 싱글 테넌트, 완전 관리형 플랫폼입니다. 각각의 Dedicated Cloud 인스턴스는 다른 W&B Dedicated Cloud 인스턴스들과 분리된 네트워크, 컴퓨트, 스토리지를 가집니다. W&B 관련 메타데이터와 데이터는 격리된 클라우드 스토리지에 저장되며, 격리된 클라우드 컴퓨트 서비스로 처리됩니다.

W&B Dedicated Cloud 는 [각 클라우드 제공업체별 다양한 글로벌 리전]({{< relref path="./dedicated_regions.md" lang="ko" >}})에서 사용할 수 있습니다.

## 데이터 보안

[Secure storage connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 이용해 [인스턴스 및 팀 단위]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ko" >}})로 직접 버킷(BYOB: Bring Your Own Bucket)을 설정하여 모델, Datasets 등 다양한 파일을 저장할 수 있습니다.

W&B Multi-tenant Cloud 와 유사하게, 여러 팀이 하나의 버킷을 공유하거나 팀별로 분리된 버킷을 사용할 수 있습니다. 팀에 Secure storage connector 를 설정하지 않으면, 해당 데이터는 인스턴스 레벨 버킷에 저장됩니다.

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="Dedicated Cloud architecture diagram" >}}

Secure storage connector 기반 BYOB 뿐 아니라, [IP allowlisting]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ko" >}})을 활용하여 신뢰할 수 있는 네트워크 위치에서만 Dedicated Cloud 인스턴스에 엑세스하도록 제한할 수 있습니다.

[클라우드 제공업체의 보안 연결 솔루션]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ko" >}})을 이용해 Dedicated Cloud 인스턴스에 프라이빗하게 연결할 수 있습니다.

배포가 조직의 정책 및 필요 시 [STIG(Sercurity Technical Implementation Guidelines)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide)를 준수하는지 확인하는 것은 사용자 책임입니다.

## Identity and access management (IAM)

W&B Organization 내에서 안전한 인증과 효과적인 권한 부여를 위해 아이덴티티 및 엑세스 관리(IAM) 기능을 제공합니다. Dedicated Cloud 인스턴스에서 제공되는 IAM 기능은 다음과 같습니다.

* [OpenID Connect(OIDC)로 SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ko" >}}) 또는 [LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ko" >}})로 인증합니다.
* 조직 단위와 팀 내에서 [사용자 역할을 적절히 설정]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ko" >}})합니다.
* [Restricted Projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})로 W&B Project 의 접근 범위를 정의하고, 누구에게 조회·수정·run 제출 권한이 있는지 제한할 수 있습니다.
* [Identity federation]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ko" >}})을 통해 JSON Web Token 을 사용해 W&B API 에 엑세스할 수 있습니다.

## 모니터링

[Audit logs]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ko" >}})을 활용해 팀 내 사용자 활동을 추적하고, 엔터프라이즈 거버넌스 요구사항을 충족할 수 있습니다. 또한 Dedicated Cloud 인스턴스 내 [W&B Organization Dashboard]({{< relref path="/guides/hosting/monitoring-usage/org_dashboard.md" lang="ko" >}})에서 조직의 사용량을 확인할 수 있습니다.

## 유지보수

W&B Multi-tenant Cloud 와 마찬가지로, Dedicated Cloud 에서는 W&B 플랫폼을 설정하고 유지하는 데의 시간과 비용 부담이 발생하지 않습니다.

Dedicated Cloud 업데이트 관리 방식은 [서버 릴리즈 프로세스]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ko" >}})를 참고하세요.

## 컴플라이언스

W&B Dedicated Cloud 에 대한 보안 통제는 내부·외부 감사를 통해 주기적으로 검증받습니다. 보안 및 컴플라이언스 문서가 필요할 경우 [W&B Security Portal](https://security.wandb.ai/)을 방문해 요청할 수 있습니다.

## 마이그레이션 옵션

[Self-Managed instance]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}}) 또는 [Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ko" >}})에서 Dedicated Cloud 로의 마이그레이션도 지원되며, 특정 한계 및 마이그레이션 관련 제약이 적용될 수 있습니다.

## 다음 단계

Dedicated Cloud 사용에 관심이 있다면 [이 양식](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)을 제출해 주세요.