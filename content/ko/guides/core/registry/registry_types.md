---
title: 레지스트리 타입
menu:
  default:
    identifier: ko-guides-core-registry-registry_types
    parent: registry
weight: 1
---

W&B는 두 가지 유형의 레지스트리를 지원합니다: [Core registries]({{< relref path="#core-registry" lang="ko" >}})와 [Custom registries]({{< relref path="#custom-registry" lang="ko" >}})입니다.

## Core registry
Core registry는 특정 유스 케이스를 위한 템플릿입니다: **Models**와 **Datasets**.

기본적으로 **Models** registry는 `"model"` 아티팩트 유형을, **Dataset** registry는 `"dataset"` 아티팩트 유형을 허용하도록 설정되어 있습니다. 관리자는 추가로 허용할 아티팩트 유형을 더할 수 있습니다.

{{< img src="/images/registry/core_registry_example.png" alt="Core registry" >}}

위 이미지는 W&B Registry 앱 UI에서 **Models**, **Dataset** core registry와, 커스텀 registry인 **Fine_Tuned_Models**를 보여줍니다.

Core registry는 [organization visibility]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ko" >}})를 가집니다. Registry 관리자는 core registry의 visibility를 변경할 수 없습니다.

## Custom registry
Custom registry는 `"model"` 아티팩트 유형이나 `"dataset"` 아티팩트 유형으로 제한되지 않습니다.

초기 데이터 수집부터 최종 모델 배포까지, 기계학습 파이프라인의 각 단계별로 custom registry를 만들 수 있습니다.

예를 들어, 학습된 모델의 성능을 평가하기 위해 선별된 데이터셋을 정리하는 용도로 "Benchmark_Datasets"라는 registry를 생성할 수 있습니다. 이 registry 내에는, 모델이 트레이닝 중 한 번도 본 적 없는 사용자 질문과 이에 대해 전문가가 검증한 답변 세트가 담긴 "User_Query_Insurance_Answer_Test_Data"와 같은 collection을 추가할 수 있습니다.

{{< img src="/images/registry/custom_registry_example.png" alt="Custom registry example" >}}

Custom registry는 [organization 또는 restricted visibility]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ko" >}})를 가질 수 있습니다. Registry 관리자는 custom registry의 visibility를 organization에서 restricted로 변경할 수 있습니다. 그러나 restricted에서 다시 organization visibility로는 변경할 수 없습니다.

Custom registry 생성 방법은 [Create a custom registry]({{< relref path="./create_collection.md" lang="ko" >}})를 참고하세요.

## Summary
아래 표는 core registry와 custom registry의 차이점을 요약한 것입니다:

|                | Core  | Custom|
| -------------- | ----- | ----- |
| Visibility     | 조직 내부 공개만 가능. 공개 범위 변경 불가. | 조직 또는 제한 공개 중 선택 가능. 조직에서 제한 공개로 변경 가능.|
| Metadata       | 미리 설정되어 있으며 사용자 수정 불가. | 사용자가 수정 가능.  |
| Artifact types | 미리 설정되어 있으며 제거할 수 없음. 사용자는 추가로 허용할 아티팩트 유형을 더할 수 있음. | 관리자가 허용할 유형을 정의할 수 있음. |
| Customization    | 기존 목록에 유형 추가 가능.|  Registry 이름, 설명, 공개 범위, 허용 아티팩트 유형 수정 가능.|