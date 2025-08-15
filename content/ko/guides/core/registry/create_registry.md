---
title: 커스텀 레지스트리 만들기
menu:
  default:
    identifier: ko-guides-core-registry-create_registry
    parent: registry
weight: 2
---

사용자 지정 레지스트리는 사용할 수 있는 아티팩트 타입에 대한 유연성과 제어를 제공하며, 레지스트리의 가시성을 제한하는 등의 다양한 기능을 지원합니다.

{{% pageinfo color="info" %}}
핵심 레지스트리와 사용자 지정 레지스트리의 전체 비교는 [Registry types]({{< relref path="registry_types.md#summary" lang="ko" >}})의 요약 표를 참고하세요.
{{% /pageinfo %}}

## 사용자 지정 레지스트리 생성하기

사용자 지정 레지스트리를 생성하려면:
1. https://wandb.ai/registry/ 의 **Registry** 앱으로 이동합니다.
2. **Custom registry** 내에서 **Create registry** 버튼을 클릭합니다.
3. **Name** 필드에 레지스트리의 이름을 입력합니다.
4. 선택적으로 레지스트리에 대한 설명을 추가할 수 있습니다.
5. **Registry visibility** 드롭다운에서 레지스트리를 볼 수 있는 대상자를 선택합니다. 자세한 가시성 옵션은 [Registry visibility types]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ko" >}})를 참고하세요.
6. **Accepted artifacts type** 드롭다운에서 **All types** 또는 **Specify types**를 선택하세요.
7. (**Specify types**를 선택한 경우) 레지스트리가 허용하는 하나 이상의 아티팩트 타입을 추가하세요.
8. **Create registry** 버튼을 클릭합니다.

{{% alert %}}
아티팩트 타입은 한 번 레지스트리의 설정에 저장되면 삭제할 수 없습니다.
{{% /alert %}}

예를 들어, 아래 이미지는 사용자가 생성하려는 `Fine_Tuned_Models`라는 사용자 지정 레지스트리를 보여줍니다. 이 레지스트리는 **Restricted**로 설정되어 있어서, 수동으로 추가된 멤버만 엑세스할 수 있습니다.

{{< img src="/images/registry/create_registry.gif" alt="새 레지스트리 생성" >}}

## 가시성 유형

레지스트리의 *가시성*은 해당 레지스트리에 누가 엑세스할 수 있는지 결정합니다. 사용자 지정 레지스트리의 가시성을 제한하면 지정된 멤버만 해당 레지스트리에 엑세스할 수 있도록 할 수 있습니다.

사용자 지정 레지스트리에서 선택할 수 있는 가시성 옵션은 두 가지입니다:

| 가시성 | 설명 |
| --- | --- |
| Restricted   | 초대받은 조직 멤버만 레지스트리에 엑세스할 수 있습니다.|
| Organization | 조직 내 모든 사용자가 레지스트리에 엑세스할 수 있습니다. |

팀 관리자 또는 레지스트리 관리자는 사용자 지정 레지스트리의 가시성을 설정할 수 있습니다.

Restricted 가시성으로 사용자 지정 레지스트리를 생성한 사용자는 자동으로 해당 레지스트리의 레지스트리 관리자(registry admin)로 추가됩니다.

## 사용자 지정 레지스트리의 가시성 설정하기

팀 관리자 또는 레지스트리 관리자는 사용자 지정 레지스트리를 만들 때나 생성 이후 언제든지 레지스트리의 가시성을 지정할 수 있습니다.

기존 사용자 지정 레지스트리의 가시성을 제한하려면:

1. https://wandb.ai/registry/ 의 **Registry** 앱으로 이동합니다.
2. 원하는 레지스트리를 선택합니다.
3. 오른쪽 상단의 기어 아이콘(설정)을 클릭합니다.
4. **Registry visibility** 드롭다운에서 원하는 가시성을 선택하세요.
5. **Restricted visibility**를 선택한 경우:
   1. 이 레지스트리에 엑세스할 조직 멤버를 추가하세요. **Registry members and roles** 항목으로 스크롤해 **Add member** 버튼을 클릭합니다.
   2. **Member** 필드에 추가할 멤버의 이메일 또는 사용자 이름을 입력합니다.
   3. **Add new member**를 클릭합니다.

{{< img src="/images/registry/change_registry_visibility.gif" alt="레지스트리 가시성 설정을 private에서 public 또는 조직 제한으로 변경하는 모습" >}}

팀 관리자가 사용자 지정 레지스트리를 만들 때 가시성을 지정하는 방법에 대해서는 [Create a custom registry]({{< relref path="./create_registry.md#create-a-custom-registry" lang="ko" >}})를 참고하세요.