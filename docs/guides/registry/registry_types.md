---
title: Registry types
displayed_sidebar: default
---

W&B는 두 가지 유형의 레지스트리를 지원합니다: [Core registries](#core-registry)와 [Custom registries](#custom-registry).

## Core registry

Core registry는 특정 유스 케이스를 위한 템플릿입니다: **Models** 및 **Datasets**.

기본적으로, **Models** 레지스트리는 `"model"` 아티팩트 유형을 수락하도록 구성되어 있으며, **Dataset** 레지스트리는 `"dataset"` 아티팩트 유형을 수락하도록 구성되어 있습니다. 관리자는 추가 허용 아티팩트 유형을 추가할 수 있습니다.

![](/images/registry/core_registry_example.png)

위 이미지는 W&B Registry App UI에서 **Models** 및 **Dataset** 코어 레지스트리와 함께 **Fine_Tuned_Models**라는 사용자 정의 레지스트리를 보여줍니다.

코어 레지스트리는 [조직 가시성](./configure_registry.md#registry-visibility-types)을 가집니다. 레지스트리 관리자는 코어 레지스트리의 가시성을 변경할 수 없습니다.

## Custom registry

Custom registries는 `"model"` 아티팩트 유형 또는 `"dataset"` 아티팩트 유형에 제한되지 않습니다.

기계학습 파이프라인의 각 단계에 대해 초기 데이터 수집부터 최종 모델 배포까지 사용자 정의 레지스트리를 생성할 수 있습니다.

예를 들어, 훈련된 모델의 성능을 평가하기 위해 관리하는 데이터셋을 조직하는 "Benchmark_Datasets"라는 레지스트리를 생성할 수 있습니다. 이 레지스트리 내에는 사용자가 제시한 질문과 모델이 트레이닝 중에 본 적이 없는 전문가 검증 답변 세트가 포함된 "User_Query_Insurance_Answer_Test_Data"라는 컬렉션이 있을 수 있습니다.

![](/images/registry/custom_registry_example.png)

Custom registry는 [조직 또는 제한된 가시성](./configure_registry.md#registry-visibility-types)을 가질 수 있습니다. 레지스트리 관리자는 custom registry의 가시성을 조직에서 제한된 가시성으로 변경할 수 있습니다. 그러나 레지스트리 관리자는 custom registry의 가시성을 제한된 가시성에서 조직 가시성으로 변경할 수 없습니다.

사용자 정의 레지스트리 생성 방법에 대한 정보는 [Create a custom registry](./create_collection.md)를 참고하세요.

## Summary

다음 표는 코어 레지스트리와 사용자 정의 레지스트리 간의 차이점을 요약합니다:

|                | Core  | Custom|
| -------------- | ----- | ----- |
| Visibility     | 조직 가시성만. 가시성을 변경할 수 없음. | 조직 또는 제한적. 가시성을 조직에서 제한적으로 변경할 수 있음.|
| Metadata       | 사전 구성되어 있으며 사용자가 수정할 수 없음. | 사용자가 수정 가능.  |
| Artifact types | 사전 구성된 허용 아티팩트 유형은 제거할 수 없음. 사용자가 추가 허용 아티팩트 유형을 추가할 수 있음. | 관리자가 허용 유형을 정의할 수 있음. |
| Customization    | 기존 목록에 추가 유형을 추가할 수 있음.| 레지스트리 이름, 설명, 가시성 및 허용 아티팩트 유형을 편집할 수 있음.|