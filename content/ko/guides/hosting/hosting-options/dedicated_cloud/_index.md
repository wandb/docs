---
title: Dedicated Cloud
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## 전용 클라우드 사용 (싱글 테넌트 SaaS)

W&B 전용 클라우드는 W&B의 AWS, GCP 또는 Azure 클라우드 계정에 배포된 싱글 테넌트의 완전 관리형 플랫폼입니다. 각 전용 클라우드 인스턴스는 다른 W&B 전용 클라우드 인스턴스와 분리된 자체 네트워크, 컴퓨팅 및 스토리지를 갖습니다. 귀하의 W&B 특정 메타데이터와 데이터는 격리된 클라우드 스토리지에 저장되고 격리된 클라우드 컴퓨팅 서비스를 사용하여 처리됩니다.

W&B 전용 클라우드는 [각 클라우드 공급자에 대해 여러 글로벌 지역에서 사용 가능]({{< relref path="./dedicated_regions.md" lang="ko" >}})합니다.

## 데이터 보안

[보안 스토리지 커넥터]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 사용하여 [인스턴스 및 Teams 수준]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ko" >}})에서 자체 버킷(BYOB)을 가져와 Models, Datasets 등과 같은 파일을 저장할 수 있습니다.

W&B 멀티 테넌트 클라우드와 유사하게 여러 Teams에 대해 단일 버킷을 구성하거나 다른 Teams에 대해 별도의 버킷을 사용할 수 있습니다. Team에 대해 보안 스토리지 커넥터를 구성하지 않으면 해당 데이터는 인스턴스 수준 버킷에 저장됩니다.

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="" >}}

보안 스토리지 커넥터를 사용한 BYOB 외에도 [IP 허용 목록]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ko" >}})을 활용하여 신뢰할 수 있는 네트워크 위치에서만 전용 클라우드 인스턴스에 대한 엑세스를 제한할 수 있습니다.

[클라우드 공급자의 보안 연결 솔루션]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ko" >}})을 사용하여 전용 클라우드 인스턴스에 비공개로 연결할 수도 있습니다.

## ID 및 엑세스 관리 (IAM)

W&B Organization에서 안전한 인증과 효과적인 권한 부여를 위해 ID 및 엑세스 관리 기능을 사용하십시오. 전용 클라우드 인스턴스에서 IAM에 사용할 수 있는 기능은 다음과 같습니다.

* [OpenID Connect (OIDC)를 사용하여 SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ko" >}}) 또는 [LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ko" >}})를 사용하여 인증합니다.
* 조직 범위 내 및 Team 내에서 [적절한 user 역할을 구성]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ko" >}})합니다.
* W&B project의 범위를 정의하여 누가 W&B Runs를 보고, 편집하고, 제출할 수 있는지 [제한된 Projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})로 제한합니다.
* [ID 페더레이션]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ko" >}})을 통해 JSON 웹 토큰을 활용하여 W&B API에 엑세스합니다.

## 모니터링

[감사 로그]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ko" >}})를 사용하여 Teams 내에서 user 활동을 추적하고 엔터프라이즈 거버넌스 요구 사항을 준수합니다. 또한 [W&B Organization 대시보드]({{< relref path="/guides/hosting/monitoring-usage/org_dashboard.md" lang="ko" >}})에서 전용 클라우드 인스턴스의 조직 사용량을 볼 수 있습니다.

## 유지 관리

W&B 멀티 테넌트 클라우드와 마찬가지로 전용 클라우드를 통해 W&B 플랫폼을 프로비저닝하고 유지 관리하는 오버헤드 및 비용이 발생하지 않습니다.

W&B가 전용 클라우드에서 업데이트를 관리하는 방법을 이해하려면 [서버 릴리스 프로세스]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ko" >}})를 참조하십시오.

## 규정 준수

W&B 전용 클라우드에 대한 보안 제어는 내부 및 외부에서 주기적으로 감사됩니다. 제품 평가 활동에 대한 보안 및 규정 준수 문서를 요청하려면 [W&B 보안 포털](https://security.wandb.ai/)을 참조하십시오.

## 마이그레이션 옵션

[자체 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}}) 또는 [멀티 테넌트 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 전용 클라우드로의 마이그레이션이 지원됩니다.

## 다음 단계

전용 클라우드 사용에 관심이 있으시면 [이 양식](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)을 제출하십시오.
