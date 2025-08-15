---
title: W&B 플랫폼
menu:
  default:
    identifier: ko-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platform은 [Core]({{< relref path="/guides/core" lang="ko" >}}), [Models]({{< relref path="/guides/models/" lang="ko" >}}), [Weave]({{< relref path="/guides/weave/" lang="ko" >}})와 같은 W&B 제품들을 지원하는 핵심 인프라, 툴, 거버넌스 구조입니다.

W&B Platform은 다음 세 가지 배포 옵션으로 제공됩니다:

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ko" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ko" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ko" >}})

아래의 책임 매트릭스는 각 배포 옵션의 주요 차이점을 정리한 표입니다:

|                                      | Multi-tenant Cloud                        | Dedicated Cloud                                                          | Customer-managed |
|--------------------------------------|-------------------------------------------|--------------------------------------------------------------------------|------------------|
| MySQL / DB 관리                      | W&B에서 완전 호스팅 및 관리                | 고객이 원하는 클라우드 또는 리전에 W&B에서 완전 호스팅 및 관리            | 고객이 직접 완전 호스팅 및 관리  |
| 오브젝트 스토리지 (S3/GCS/Blob 스토리지) | **옵션 1**: W&B에서 완전 호스팅<br />**옵션 2**: 각 팀별로 [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 사용하여 직접 버킷 설정 가능  | **옵션 1**: W&B에서 완전 호스팅<br />**옵션 2**: 인스턴스 또는 팀 단위로 [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 사용하여 직접 버킷 설정 가능 | 고객이 직접 완전 호스팅 및 관리 |
| SSO 지원                             | W&B(에서) Auth0 를 이용해 관리             | **옵션 1**: 고객이 직접 관리<br />**옵션 2**: W&B에서 Auth0로 관리         | 고객이 직접 완전 관리    |
| W&B 서비스 (앱)                      | W&B에서 완전 관리                          | W&B에서 완전 관리                                                         | 고객이 직접 완전 관리          |
| 앱 보안                              | W&B에서 완전 관리                          | W&B와 고객이 공동 책임                                                    | 고객이 직접 완전 관리          |
| 유지보수(업그레이드, 백업 등)         | W&B에서 관리                               | W&B에서 관리                                                              | 고객이 직접 관리        |
| 지원                                 | 지원 SLA                                  | 지원 SLA                                                                 | 지원 SLA           |
| 지원 클라우드 인프라                  | GCP                                       | AWS, GCP, Azure                                                          | AWS, GCP, Azure, 온프레미스 bare-metal |

## 배포 옵션 안내

아래에서 각 배포 유형에 대한 개요를 확인할 수 있습니다.

### W&B Multi-tenant Cloud

W&B Multi-tenant Cloud는 W&B의 클라우드 인프라에서 제공되는 완전 매니지드 서비스로, 원하는 규모로 W&B 제품에 간편하게 엑세스할 수 있습니다. 가격 효율성을 갖추고 최신 기능을 지속적으로 업데이트 받으실 수 있습니다. 보안이 중요한 프라이빗 배포가 꼭 필요하지 않고, 셀프 서비스 온보딩과 비용 효율성이 중요하다면 프로덕션 AI 워크플로우나 제품 트라이얼에 Multi-tenant Cloud를 사용하는 것을 권장합니다.

자세한 내용은 [W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ko" >}})를 참고하세요.

### W&B Dedicated Cloud

W&B Dedicated Cloud는 W&B의 클라우드 인프라에서 싱글 테넌트로 제공되는 완전 매니지드 서비스입니다. 데이터 레지던시 등 엄격한 거버넌스 준수가 필요하거나, 고급 보안 옵션이 요구되며, 인프라 구축 및 운영의 복잡성 없이 보안/확장성/성능 요구 조건을 충족하면서 AI 운영 비용을 최적화하고 싶은 조직에 최적의 선택입니다.

자세한 내용은 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})를 참고하세요.

### W&B Customer-Managed

이 옵션을 사용하면, 직접 관리하는 인프라에 W&B Server를 배포해 운영할 수 있습니다. W&B Server는 W&B Platform과 그 지원 제품을 실행할 수 있는 자체 포함 패키지 방식입니다. 모든 인프라가 온프레미스에 있거나, W&B Dedicated Cloud로 충족되지 않는 엄격한 규제 준수가 필요한 경우 직접 관리형(Self-Managed)을 권장합니다. 이 옵션은 인프라 준비부터 지속적인 유지보수 및 업그레이드까지 모두 사용자가 직접 책임집니다.

자세한 내용은 [W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}})를 참고하세요.

## 다음 단계

W&B의 제품을 체험해보고 싶다면 [Multi-tenant Cloud](https://wandb.ai/home)를 이용해 보세요. 엔터프라이즈 환경에 맞는 설정을 원하신다면, 트라이얼 시작을 위한 적절한 배포 옵션을 [여기](https://wandb.ai/site/enterprise-trial)에서 선택하세요.