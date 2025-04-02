---
title: W&B Platform
menu:
  default:
    identifier: ko-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platform은 [Core]({{< relref path="/guides/core" lang="ko" >}}), [Models]({{< relref path="/guides/models/" lang="ko" >}}) 및 [Weave]({{< relref path="/guides/weave/" lang="ko" >}})와 같은 W&B 제품을 지원하는 기본적인 인프라, 툴링 및 거버넌스 스캐폴딩입니다.

W&B Platform은 세 가지의 다양한 배포 옵션으로 제공됩니다:

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ko" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ko" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ko" >}})

다음 책임 매트릭스는 주요 차이점을 간략하게 설명합니다:

|                                      | Multi-tenant Cloud                | Dedicated Cloud                                                     | Customer-managed |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------|------------------|
| MySQL / DB 관리                | W&B에서 완전 호스팅 및 관리     | W&B에서 클라우드 또는 고객 선택 지역에서 완전 호스팅 및 관리 | 고객이 완전 호스팅 및 관리 |
| 오브젝트 스토리지 (S3/GCS/Blob storage) | **옵션 1**: W&B에서 완전 호스팅<br />**옵션 2**: 고객은 [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 사용하여 팀별로 자체 버킷을 구성할 수 있습니다.  | **옵션 1**: W&B에서 완전 호스팅<br />**옵션 2**: 고객은 [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 사용하여 인스턴스 또는 팀별로 자체 버킷을 구성할 수 있습니다. | 고객이 완전 호스팅 및 관리 |
| SSO 지원                          | Auth0를 통해 W&B에서 관리             | **옵션 1**: 고객 관리<br />**옵션 2**: Auth0를 통해 W&B에서 관리 | 고객이 완전 관리   |
| W&B Service (App)                    | W&B에서 완전 관리              | W&B에서 완전 관리                                                | 고객이 완전 관리          |
| 앱 보안                         | W&B에서 완전 관리              | W&B와 고객의 공동 책임                           | 고객이 완전 관리         |
| 유지 관리 (업그레이드, 백업 등)    | W&B에서 관리 | W&B에서 관리 | 고객이 관리 |
| 지원                              | 지원 SLA                       | 지원 SLA                                                         | 지원 SLA |
| 지원되는 클라우드 인프라       | GCP                               | AWS, GCP, Azure                                                     | AWS, GCP, Azure, On-Prem bare-metal |

## 배포 옵션
다음 섹션에서는 각 배포 유형에 대한 개요를 제공합니다.

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud는 W&B의 클라우드 인프라에 배포된 완전 관리형 서비스로, 원하는 규모로 W&B 제품에 원활하게 엑세스하고, 비용 효율적인 가격 옵션을 활용하며, 최신 기능 및 업데이트를 지속적으로 받을 수 있습니다. 프라이빗 배포의 보안이 필요하지 않고, 셀프 서비스 온보딩이 중요하며, 비용 효율성이 중요한 경우 프로덕션 AI 워크플로우를 관리하기 위해 Multi-tenant Cloud를 사용하는 것이 좋습니다.

자세한 내용은 [W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ko" >}})를 참조하십시오.

### W&B Dedicated Cloud
W&B Dedicated Cloud는 W&B의 클라우드 인프라에 배포된 싱글 테넌트, 완전 관리형 서비스입니다. 데이터 상주를 포함한 엄격한 거버넌스 제어를 준수해야 하고, 고급 보안 기능이 필요하며, 보안, 확장성 및 성능 특성을 갖춘 필요한 인프라를 구축 및 관리하지 않고도 AI 운영 비용을 최적화하려는 경우 W&B를 온보딩하는 데 가장 적합합니다.

자세한 내용은 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})를 참조하십시오.

### W&B Customer-Managed
이 옵션을 사용하면 자체 관리형 인프라에 W&B Server를 배포하고 관리할 수 있습니다. W&B Server는 W&B Platform 및 지원되는 W&B 제품을 실행하기 위한 자체 포함 패키지 메커니즘입니다. 기존 인프라가 모두 온프레미스에 있거나, W&B Dedicated Cloud에서 충족되지 않는 엄격한 규제 요구 사항이 있는 경우 이 옵션을 사용하는 것이 좋습니다. 이 옵션을 사용하면 W&B Server를 지원하는 데 필요한 인프라의 프로비저닝, 지속적인 유지 관리 및 업그레이드를 완전히 책임져야 합니다.

자세한 내용은 [W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}})를 참조하십시오.

## 다음 단계

W&B 제품을 사용해 보려면 [Multi-tenant Cloud](https://wandb.ai/home)를 사용하는 것이 좋습니다. 엔터프라이즈 친화적인 설정을 찾고 있다면 [여기](https://wandb.ai/site/enterprise-trial)에서 트라이얼에 적합한 배포 유형을 선택하십시오.
