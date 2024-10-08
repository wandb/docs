---
title: Use W&B Dedicated Cloud, the Single-Tenant SaaS
displayed_sidebar: default
---

# 전용 클라우드 (싱글 테넌트 SaaS)

W&B 전용 클라우드는 W&B의 AWS, GCP 또는 Azure 클라우드 계정에 배포되는 싱글 테넌트, 완전 관리형 플랫폼입니다. 각 전용 클라우드 인스턴스는 다른 W&B 전용 클라우드 인스턴스와는 독립된 네트워크, 컴퓨팅 및 저장공간을 가집니다. W&B 관련 메타데이터와 데이터는 격리된 클라우드 저장소에 저장되며 격리된 클라우드 컴퓨트 서비스를 사용하여 처리됩니다.

W&B 전용 클라우드는 [각 클라우드 제공업체의 여러 글로벌 지역에서 사용 가능합니다](./dedicated_regions.md)

## 데이터 보안

모델, 데이터셋, 기타 파일을 저장하기 위해 [보안 저장소 커넥터](../data-security/secure-storage-connector.md)를 사용하여 [인스턴스 및 팀 수준에서](../data-security/secure-storage-connector.md#configuration-options) 자신의 버킷을 사용할 수 있습니다 (BYOB).

W&B 멀티 테넌트 클라우드와 마찬가지로, 여러 팀을 위해 하나의 버킷을 설정하거나 다른 팀을 위해 별도의 버킷을 사용할 수 있습니다. 팀에 보안 저장소 커넥터를 구성하지 않으면 해당 데이터는 인스턴스 레벨의 버킷에 저장됩니다.

![](/images/hosting/dedicated_cloud_arch.png)

보안 저장소 커넥터를 사용한 BYOB 외에도, [IP 허용 목록](../data-security/ip-allowlisting.md)을 이용하여 신뢰할 수 있는 네트워크 위치에서만 전용 클라우드 인스턴스에 엑세스를 제한할 수 있습니다.

또한 [클라우드 제공업체의 보안 연결 솔루션](../data-security/private-connectivity.md)을 사용하여 전용 클라우드 인스턴스에 비공개로 연결할 수 있습니다.

## ID 및 엑세스 관리 (IAM)

W&B 조직에서의 안전한 인증과 효과적인 권한 부여를 위해 ID 및 엑세스 관리 기능을 사용하십시오. 전용 클라우드 인스턴스에서 IAM에 대해 다음 기능이 제공됩니다:

* [OIDC를 사용한 SSO](../iam/sso.md) 또는 [LDAP](../iam/ldap.md)으로 인증합니다.
* 조직의 범위와 팀 내에서 적절한 사용자 역할을 [설정](../iam/manage-users.md)합니다.
* [제한된 프로젝트](../iam/restricted-projects.md)를 통해 W&B 프로젝트의 범위를 정의하여 누가 W&B runs를 보고, 편집하고, 제출할 수 있는지를 제한합니다.
* W&B APIs에 엑세스하기 위해 [Identity Federation](../iam/identity_federation.md)을 통한 JSON 웹 토큰을 활용하십시오.

## 모니터링

팀 내 사용자 활동을 추적하고 기업 가버넌스 요구 사항을 충족시키기 위해 [감사 로그](../monitoring-usage/audit-logging.md)를 사용하십시오. 또한, 전용 클라우드 인스턴스에서 W&B 조직 대시보드로 조직 사용량을 확인할 수 있습니다.

## 유지보수

W&B 멀티 테넌트 클라우드와 유사하게, 전용 클라우드를 사용하면 W&B 플랫폼을 프로비저닝하고 유지 관리하는 것에 따른 오버헤드와 비용이 발생하지 않습니다.

W&B가 전용 클라우드에서 업데이트를 관리하는 방식을 이해하려면 [서버 릴리스 프로세스](../server-release-process.md)를 참조하십시오.

## 준수

W&B 전용 클라우드의 보안 통제는 주기적으로 내부 및 외부에서 감사됩니다. 귀사의 제품 평가 연습을 위해 보안 및 준수 문서를 요청하려면 [W&B 보안 포털](https://security.wandb.ai/)을 참조하십시오.

## 마이그레이션 옵션

[셀프 매니지드 인스턴스](./self-managed.md) 또는 [멀티 테넌트 클라우드](./saas_cloud.md)에서 전용 클라우드로의 마이그레이션이 지원됩니다.

## 다음 단계

전용 클라우드를 사용하고자 한다면 [이 양식](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)을 제출하십시오.