---
description: Deploying W&B in production
displayed_sidebar: default
---

# 자체 관리 호스팅

:::info
W&B 서버를 인프라에 개인적으로 호스팅하기 전에 [W&B 관리 호스팅 옵션](./wb-managed.md)을 고려하는 것이 좋습니다. W&B 클라우드는 간단하고 안전하며, 최소한의 구성만 필요합니다.
:::

## 온-프레미스 프라이빗 클라우드

온-프레미스 프라이빗 클라우드는 고객의 프라이빗 클라우드 인프라에서 확장 가능한 배포로 W&B 서버가 실행되는 완전 자체 호스팅 솔루션입니다. W&B는 고객이 AWS/GCP/Azure에 배포하기 위해 공식 W&B 테라폼 스크립트를 사용할 것을 권장합니다. 고객은 모든 W&B 서비스가 제공되는 원하는 지역에 이를 배포할 수 있습니다. 환경은 우리 또는 귀사의 IT/DevOps/MLOps 팀이 테라폼과 쿠버네티스로 구성된 도구 집합을 사용하여 프로비저닝할 수 있습니다. 인스턴스의 업그레이드 및 유지 관리는 고객의 IT/DevOps/MLOps 팀이 처리해야 합니다.

인프라를 구성하는 가장 간단한 방법은 W&B의 공식 테라폼 스크립트를 사용하는 것입니다:

- [Amazon Web Services (AWS)](https://github.com/wandb/terraform-aws-wandb)
- [Google Cloud Platform (GCP)](https://github.com/wandb/terraform-google-wandb)
- [Microsoft Azure](https://github.com/wandb/terraform-azurerm-wandb)

## 온-프레미스 베어 메탈

이것은 고객의 온-프레미스 베어 메탈 인프라에서 확장 가능한 배포로 W&B 서버가 실행되는 완전 자체 호스팅 솔루션입니다. 온-프레미스 베어 메탈 설치에서 W&B 서버를 설정하고 구성하기 위해 필요한 여러 인프라 구성 요소가 있으며 이에는 다음이 포함되지만 이에 국한되지 않습니다

- 완전히 확장 가능한 MySQL 8 데이터베이스
- S3 호환 개체 저장소
- 메시지 큐
- 레디스 캐시 (선택 사항)

W&B는 호환 가능한 데이터베이스 엔진, 개체 저장소에 대한 추천을 제공하며 설치 프로세스를 도울 경험이 풍부한 팀이 있습니다. 데이터베이스를 관리하고 분산 개체 저장 시스템을 생성 및 유지 관리하는 복잡성은 고객의 IT/DevOps/MLOps 팀에 추가적인 부담을 줍니다. 가능할 때 W&B는 사용자 경험을 향상시키기 위해 W&B 관리 클라우드 솔루션을 사용할 것을 권장합니다.

### 연락처

W&B의 온-프레미스 설치 계획에 대해 궁금한 사항이 있으면 [온-프레미스 / 베어메탈](../how-to-guides/bare-metal.md) 문서를 참조하고 support@wandb.com으로 W&B 지원팀에 문의하십시오.