---
title: Document machine learning model
description: 모델 카드에 설명을 추가하여 모델을 문서화하세요.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-create-model-cards
    parent: model-registry
weight: 8
---

등록된 모델의 모델 카드에 설명을 추가하여 머신러닝 모델의 여러 측면을 문서화하세요. 문서화할 가치가 있는 몇 가지 주제는 다음과 같습니다.

* **요약**: 모델에 대한 요약입니다. 모델의 목적, 모델이 사용하는 머신러닝 프레임워크 등입니다.
* **트레이닝 데이터**: 사용된 트레이닝 데이터, 트레이닝 데이터 세트에 대해 수행된 처리, 해당 데이터가 저장된 위치 등을 설명합니다.
* **아키텍처**: 모델 아키텍처, 레이어 및 특정 설계 선택에 대한 정보입니다.
* **모델 역직렬화**: 팀 구성원이 모델을 메모리에 로드하는 방법에 대한 정보를 제공합니다.
* **Task**: 머신러닝 모델이 수행하도록 설계된 특정 유형의 Task 또는 문제입니다. 모델의 의도된 기능을 분류한 것입니다.
* **라이선스**: 머신러닝 모델 사용과 관련된 법적 조건 및 권한입니다. 이를 통해 모델 사용자는 모델을 활용할 수 있는 법적 프레임워크를 이해할 수 있습니다.
* **참조**: 관련 연구 논문, 데이터셋 또는 외부 리소스에 대한 인용 또는 참조입니다.
* **배포**: 모델이 배포되는 방식 및 위치에 대한 세부 정보와 워크플로우 오케스트레이션 플랫폼과 같은 다른 엔터프라이즈 시스템에 모델을 통합하는 방법에 대한 지침입니다.

## 모델 카드에 설명 추가

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)의 W&B Model Registry 앱으로 이동합니다.
2. 모델 카드를 생성하려는 등록된 모델 이름 옆에 있는 **세부 정보 보기**를 선택합니다.
2. **Model card** 섹션으로 이동합니다.

{{< img src="/images/models/model_card_example.png" alt="" >}}

3. **Description** 필드 내에 머신러닝 모델에 대한 정보를 제공합니다. [Markdown 마크업 언어](https://www.markdownguide.org/)를 사용하여 모델 카드 내에서 텍스트 서식을 지정합니다.

예를 들어 다음 이미지는 **신용카드 채무 불이행 예측** 등록 모델의 모델 카드를 보여줍니다.

{{< img src="/images/models/model_card_credit_example.png" alt="" >}}
