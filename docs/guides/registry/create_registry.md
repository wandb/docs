---
title: Create a custom registry
displayed_sidebar: default
---

[사용자 정의 레지스트리](./registry_types.md#custom-registry)를 만들어서 ML 워크플로우의 각 단계를 관리하세요.

사용자 정의 레지스트리는 기본 [핵심 레지스트리](./registry_types.md#core-registry)와 다른 프로젝트별 요구 사항을 조직하는 데 특히 유용합니다.

다음 절차에서는 대화형으로 레지스트리를 만드는 방법을 설명합니다:
1. W&B App UI에서 **Registry** 앱으로 이동합니다.
2. **Custom registry** 내에서 **Create registry** 버튼을 클릭합니다.
3. **Name** 필드에 레지스트리의 이름을 입력합니다.
4. 선택적으로 레지스트리에 대한 설명을 입력합니다.
5. **Registry visibility** 드롭다운에서 누가 이 레지스트리를 볼 수 있는지 선택합니다. 레지스트리 보기 옵션에 대한 자세한 내용은 [레지스트리 보기 유형](./configure_registry.md#registry-visibility-types)을 참조하세요.
6. **Accepted artifacts type** 드롭다운에서 **All types** 또는 **Specify types** 중 하나를 선택합니다.
7. (**Specify types**을 선택한 경우) 레지스트리가 수용하는 하나 이상의 아티팩트 유형을 추가합니다.
:::안내
아티팩트 유형은 한 번 추가되어 레지스트리의 설정에 저장되면 제거할 수 없습니다.
:::
8. **Create registry** 버튼을 클릭합니다.

![](/images/registry/create_registry.gif)

예를 들어, 앞의 이미지는 "Fine_Tuned_Models"라는 사용자 정의 레지스트리를 만들기 직전의 모습입니다. 이 레지스트리는 **Restricted**로 설정되어 있어 "Fine_Tuned_Models" 레지스트리에 수동으로 추가된 멤버들만 이 레지스트리에 엑세스할 수 있습니다.