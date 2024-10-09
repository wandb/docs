---
title: W&B Platform
slug: /guides/hosting
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Platform은 [Core](../core.md), [Models](../models.md), [Weave](../weave_platform.md)와 같은 W&B 제품을 지원하는 기본 인프라, 툴링 및 거버넌스 구조입니다.

W&B Platform은 세 가지 배포 옵션으로 제공됩니다:

* [W&B Multi-tenant Cloud](#wb-multi-tenant-cloud)
* [W&B Dedicated Cloud](#wb-dedicated-cloud)
* [W&B Customer-managed](#wb-customer-managed)

다음의 책임 매트릭스는 각 옵션 간의 주요 차이점을 설명합니다:
![](/images/hosting/shared_responsibility_matrix.png)

## 배포 옵션
다음 섹션에서는 각 유형의 배포에 대한 개요를 제공합니다.

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud는 W&B의 클라우드 인프라에 배포된 완전 관리형 서비스로, 원하는 규모로 W&B 제품에 원활하게 엑세스할 수 있으며, 비용 효율적인 가격 옵션과 최신 기능 및 성능에 대한 지속적인 업데이트를 제공합니다. W&B는 Multi-tenant Cloud를 제품 트라이얼, 혹은 개인 배포의 보안이 필요하지 않고, 셀프 서비스 온보딩이 중요하며, 비용 효율성이 중요한 경우 프로덕션 AI 워크플로우 관리를 위해 사용할 것을 권장합니다.

자세한 내용은 [W&B Multi-tenant Cloud](./hosting-options/saas_cloud.md)를 참조하십시오.

### W&B Dedicated Cloud
W&B Dedicated Cloud는 W&B의 클라우드 인프라에 배포된 싱글 테넌트, 완전 관리형 서비스입니다. 엄격한 거버넌스 통제를 포함한 데이터 거주 요건 준수가 필요하고, 보안 기능이 강화된 환경을 필요로 하며, 보안, 스케일, 성능 특징이 있는 필요한 인프라를 구축하고 관리할 필요 없이 AI 운영 비용을 최적화하려는 조직에 적합한 장소입니다.

자세한 내용은 [W&B Dedicated Cloud](./hosting-options/dedicated_cloud.md)를 참조하십시오.

### W&B Customer-Managed
이 옵션에서는 W&B Server를 직접 관리하는 인프라에 배포하고 관리할 수 있습니다. W&B Server는 W&B Platform 및 지원되는 W&B 제품을 run할 수 있는 자체 포함된 패키지 메커니즘입니다. 기존의 모든 인프라가 온프레미스에 있거나, W&B Dedicated Cloud로 만족되지 않는 엄격한 규제 요건을 가진 조직인 경우 이 옵션을 추천드립니다. 이 옵션을 통해 W&B Server를 지원하기 위해 필요한 인프라의 프로비저닝 및 지속적인 유지보수 및 업그레이드를 관리할 모든 책임을 갖게 됩니다.

자세한 내용은 [W&B Self Managed](./hosting-options/self-managed.md)를 참조하십시오.

## 다음 단계

W&B 제품을 사용해 보고 싶다면, [Multi-tenant Cloud](https://wandb.ai/home)를 사용하는 것을 권장합니다. 기업 친화적인 설정을 원하신다면, 트라이얼을 위한 적절한 배포 유형을 [여기](https://wandb.ai/site/enterprise-trial)에서 선택하세요.