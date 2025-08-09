---
title: 기계학습 모델 문서화
description: 모델 카드를 통해 모델을 문서화하고 설명을 추가하세요.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-create-model-cards
    parent: model-registry
weight: 8
---

등록된 모델의 모델 카드에 설명을 추가하여 기계학습 모델의 다양한 측면을 문서화하세요. 아래와 같은 주제들을 문서화하는 것이 좋습니다:

* **요약**: 모델이 무엇인지에 대한 요약과 모델의 목적, 사용하는 기계학습 프레임워크 등.
* **트레이닝 데이터**: 사용된 트레이닝 데이터, 트레이닝 데이터셋에 적용된 처리 과정, 데이터가 어디에 저장되어 있는지 등 설명.
* **아키텍처**: 모델 아키텍처, 레이어 구성, 특정 설계 선택 등에 대한 정보.
* **모델 역직렬화**: 팀원이 해당 모델을 메모리에 로드할 수 있는 방법에 대한 안내.
* **태스크**: 해당 기계학습 모델이 수행하도록 설계된 특정 태스크 또는 문제 유형. 모델의 주요 역량을 범주화하는 항목.
* **라이선스**: 기계학습 모델 사용과 관련된 법적 조건 및 권한. 사용자가 어떤 법적 프레임워크에서 모델을 활용할 수 있는지 이해하는 데 도움을 줍니다.
* **참고 자료**: 관련 연구 논문, 데이터셋, 외부 자료에 대한 인용 및 참고 문헌.
* **배포**: 모델이 어떻게, 어디에 배포되는지, 워크플로우 오케스트레이션 플랫폼 등과 같은 다른 엔터프라이즈 시스템과 통합하는 방법에 대한 안내.

## 모델 카드에 설명 추가하기

1. [W&B Model Registry 앱](https://wandb.ai/registry/model)으로 이동합니다.
2. 모델 카드를 생성하려는 등록된 모델 이름 옆의 **View details**를 선택합니다.
2. **Model card** 섹션으로 이동합니다.
{{< img src="/images/models/model_card_example.png" alt="Model card example" >}}
3. **Description** 필드에 기계학습 모델에 대한 정보를 입력합니다. 모델 카드의 텍스트는 [Markdown 마크업 언어](https://www.markdownguide.org/)로 포맷할 수 있습니다.

예를 들어, 아래 이미지는 **Credit-card Default Prediction** 등록된 모델의 모델 카드 예시입니다.
{{< img src="/images/models/model_card_credit_example.png" alt="Model card credit scoring" >}}