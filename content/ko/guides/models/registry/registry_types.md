---
title: Registry types
menu:
  default:
    identifier: ko-guides-models-registry-registry_types
    parent: registry
weight: 1
---

W\&B는 두 가지 유형의 레지스트리를 지원합니다. [코어 레지스트리]({{< relref path="#core-registry" lang="ko" >}})와 [커스텀 레지스트리]({{< relref path="#custom-registry" lang="ko" >}})가 있습니다.

## 코어 레지스트리
코어 레지스트리는 특정 유스 케이스를 위한 템플릿입니다. **Models** 및 **Datasets**가 있습니다.

기본적으로 **Models** 레지스트리는 `"model"` 아티팩트 유형을 허용하도록 구성되어 있고 **Dataset** 레지스트리는 `"dataset"` 아티팩트 유형을 허용하도록 구성되어 있습니다. 관리자는 허용되는 아티팩트 유형을 추가할 수 있습니다.

{{< img src="/images/registry/core_registry_example.png" alt="" >}}

위의 이미지는 W\&B Registry App UI에서 **Models**, **Dataset** 코어 레지스트리와 **Fine_Tuned_Models**라는 커스텀 레지스트리를 보여줍니다.

코어 레지스트리는 [organization visibility]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ko" >}})를 가집니다. 레지스트리 관리자는 코어 레지스트리의 visibility를 변경할 수 없습니다.

## 커스텀 레지스트리
커스텀 레지스트리는 `"model"` 아티팩트 유형 또는 `"dataset"` 아티팩트 유형으로 제한되지 않습니다.

초기 데이터 수집부터 최종 모델 배포에 이르기까지 기계 학습 파이프라인의 각 단계에 대한 커스텀 레지스트리를 만들 수 있습니다.

예를 들어, 트레이닝된 모델의 성능을 평가하기 위해 큐레이팅된 데이터셋을 구성하기 위해 "Benchmark_Datasets"라는 레지스트리를 만들 수 있습니다. 이 레지스트리 내에는 모델이 트레이닝 중에 본 적이 없는 사용자 질문과 해당 전문가 검증 답변 세트가 포함된 "User_Query_Insurance_Answer_Test_Data"라는 컬렉션이 있을 수 있습니다.

{{< img src="/images/registry/custom_registry_example.png" alt="" >}}

커스텀 레지스트리는 [organization 또는 restricted visibility]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ko" >}})를 가질 수 있습니다. 레지스트리 관리자는 커스텀 레지스트리의 visibility를 organization에서 restricted로 변경할 수 있습니다. 그러나 레지스트리 관리자는 커스텀 레지스트리의 visibility를 restricted에서 organization visibility로 변경할 수 없습니다.

커스텀 레지스트리를 만드는 방법에 대한 자세한 내용은 [커스텀 레지스트리 만들기]({{< relref path="./create_collection.md" lang="ko" >}})를 참조하세요.

## 요약
다음 표는 코어 및 커스텀 레지스트리 간의 차이점을 요약한 것입니다.

|                | Core  | Custom|
| -------------- | ----- | ----- |
| Visibility     | Organizational visibility만 해당됩니다. Visibility를 변경할 수 없습니다. | organization 또는 restricted 중 하나입니다. Visibility를 organization에서 restricted visibility로 변경할 수 있습니다.|
| Metadata       | 사전 구성되어 있으며 사용자가 편집할 수 없습니다. | 사용자가 편집할 수 있습니다.  |
| Artifact types | 사전 구성되어 있으며 허용되는 아티팩트 유형을 제거할 수 없습니다. 사용자는 허용되는 아티팩트 유형을 추가할 수 있습니다. | 관리자는 허용되는 유형을 정의할 수 있습니다. |
| Customization    | 기존 목록에 유형을 추가할 수 있습니다.|  레지스트리 이름, 설명, visibility 및 허용되는 아티팩트 유형을 편집합니다.|
