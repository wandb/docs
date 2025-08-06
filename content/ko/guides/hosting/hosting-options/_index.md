---
title: 배포 옵션
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-_index
    parent: w-b-platform
weight: 1
---

이 섹션에서는 W&B 를 배포할 수 있는 다양한 방법을 설명합니다.

## W&B 멀티 테넌트 클라우드
[W&B 멀티 테넌트 클라우드]({{< relref path="saas_cloud.md" lang="ko" >}})는 업그레이드, 유지관리, 플랫폼 보안, 용량 계획 등 모든 부분이 W&B 에 의해 완전히 관리됩니다. 멀티 테넌트 클라우드는 W&B 의 Google Cloud Platform(GCP) 계정의 [GCP 북미 리전](https://cloud.google.com/compute/docs/regions-zones)에 배포됩니다. [Bring your own bucket (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 기능을 사용하면 W&B Artifacts 및 기타 민감한 데이터를 사용자의 클라우드 또는 온프레미스 인프라에 저장할 수도 있습니다.

자세한 내용은 [W&B 멀티 테넌트 클라우드]({{< relref path="saas_cloud.md" lang="ko" >}})를 참고하시거나 [무료로 시작하기](https://app.wandb.ai/login?signup=true)를 눌러보세요.

## W&B 전용 클라우드
[W&B 전용 클라우드]({{< relref path="dedicated_cloud/" lang="ko" >}})는 엔터프라이즈 조직을 위해 설계된 싱글 테넌트, 완전 관리형 플랫폼입니다. W&B 전용 클라우드는 W&B 의 AWS, GCP, 또는 Azure 계정에 배포됩니다. 전용 클라우드는 멀티 테넌트 클라우드보다 더 많은 유연성을 제공하며, W&B Self-Managed 보다는 복잡성이 낮습니다. 업그레이드, 유지관리, 플랫폼 보안, 용량 계획은 모두 W&B 가 관리합니다. 각 Dedicated Cloud 인스턴스는 다른 W&B Dedicated Cloud 인스턴스와 완전히 분리된 네트워크, 컴퓨트, 스토리지를 가집니다.

W&B 관련 메타데이터와 데이터는 분리된 클라우드 스토리지에 저장되며, 분리된 클라우드 컴퓨트 서비스를 통해 처리됩니다. [Bring your own bucket (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 기능을 사용하면, Artifacts 및 기타 민감한 데이터를 사용자의 클라우드나 온프레미스 인프라에 저장할 수 있습니다.

W&B 전용 클라우드에는 중요한 보안 및 다양한 엔터프라이즈 기능을 지원하는 [엔터프라이즈 라이선스]({{< relref path="self-managed/server-upgrade-process.md" lang="ko" >}})가 포함되어 있습니다.

높은 수준의 보안 또는 컴플라이언스 요건을 가진 조직의 경우, HIPAA 준수, Single Sign On, Customer Managed Encryption Keys (CMEK)와 같은 기능을 **엔터프라이즈** 지원과 함께 제공합니다. [추가 정보 요청하기](https://wandb.ai/site/contact).

자세한 내용은 [W&B 전용 클라우드]({{< relref path="dedicated_cloud/" lang="ko" >}})를 참고하시거나 [무료로 시작하기](https://app.wandb.ai/login?signup=true)를 눌러보세요.

## W&B Self-Managed
[W&B Self-Managed]({{< relref path="self-managed/" lang="ko" >}})는 직접 관리하는 형태로, 사용자의 온프레미스 또는 직접 관리하는 클라우드 인프라에 배포합니다. IT/DevOps/MLOps 팀은 다음을 책임집니다:
- 배포 환경 프로비저닝
- 기관의 정책 및 필요에 따라 [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) 준수를 포함한 인프라 보안
- 업그레이드 및 패치 적용
- Self-Managed W&B Server 인스턴스의 지속적인 유지관리

W&B Self-Managed 용 엔터프라이즈 라이선스를 선택적으로 구매할 수 있습니다. 엔터프라이즈 라이선스에는 중요한 보안 및 다양한 엔터프라이즈 기능 지원이 포함되어 있습니다.

자세한 내용은 [W&B Self-Managed]({{< relref path="self-managed/" lang="ko" >}}) 페이지를 확인하거나, [참조 아키텍처]({{< relref path="self-managed/ref-arch.md" lang="ko" >}}) 가이드를 참고하세요.