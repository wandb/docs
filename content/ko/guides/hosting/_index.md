---
title: W&B Platform
menu:
  default:
    identifier: ko-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platform은 [Core]({{< relref path="/guides/core/" lang="ko" >}}), [Models]({{< relref path="/guides/models/" lang="ko" >}}) 및 [Weave]({{< relref path="/guides/weave/" lang="ko" >}})와 같은 W&B 제품을 지원하는 기본적인 인프라, 툴링 및 거버넌스 Scaffolding입니다.

W&B Platform은 세 가지 배포 옵션으로 제공됩니다.

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ko" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ko" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ko" >}})

다음 책임 매트릭스는 다양한 옵션 간의 주요 차이점을 간략하게 설명합니다.
{{< img src="/images/hosting/shared_responsibility_matrix.png" alt="" >}}

## 배포 옵션
다음 섹션에서는 각 배포 유형에 대한 개요를 제공합니다.

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud는 W&B의 클라우드 인프라에 배포된 완전 관리형 서비스로, 원하는 규모로 W&B 제품에 원활하게 엑세스하고, 비용 효율적인 가격 옵션을 이용하며, 최신 기능에 대한 지속적인 업데이트를 받을 수 있습니다. W&B는 제품 트라이얼에 Multi-tenant Cloud를 사용하거나, 개인 배포의 보안이 필요하지 않고, 셀프 서비스 온보딩이 중요하며, 비용 효율성이 중요한 경우 프로덕션 AI 워크플로우를 관리하는 데 Multi-tenant Cloud를 사용하는 것을 권장합니다.

자세한 내용은 [W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ko" >}})를 참조하세요.

### W&B Dedicated Cloud
W&B Dedicated Cloud는 W&B의 클라우드 인프라에 배포된 싱글 테넌트 완전 관리형 서비스입니다. 데이터 상주를 포함한 엄격한 거버넌스 제어 준수가 필요하고, 고급 보안 기능이 필요하며, 보안, 확장성 및 성능 특성을 갖춘 필요한 인프라를 구축 및 관리하지 않고도 AI 운영 비용을 최적화하려는 경우 W&B를 온보딩하기에 가장 적합한 곳입니다.

자세한 내용은 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})를 참조하세요.

### W&B Customer-Managed
이 옵션을 사용하면 자체 관리형 인프라에 W&B Server를 배포하고 관리할 수 있습니다. W&B Server는 W&B Platform 및 지원되는 W&B 제품을 실행하는 자체 포함된 패키지 메커니즘입니다. 기존 인프라가 모두 온프레미스에 있거나, W&B Dedicated Cloud로 충족되지 않는 엄격한 규제 요구 사항이 있는 경우 이 옵션을 권장합니다. 이 옵션을 사용하면 W&B Server를 지원하는 데 필요한 인프라의 프로비저닝, 지속적인 유지 관리 및 업그레이드를 관리할 책임이 있습니다.

자세한 내용은 [W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}})를 참조하세요.

## 다음 단계

W&B 제품을 사용해 보려면 [Multi-tenant Cloud](https://wandb.ai/home)를 사용하는 것이 좋습니다. 엔터프라이즈 친화적인 설정을 찾고 있다면 [여기](https://wandb.ai/site/enterprise-trial)에서 트라이얼에 적합한 배포 유형을 선택하세요.
