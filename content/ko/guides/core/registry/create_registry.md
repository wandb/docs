---
title: Create a custom registry
menu:
  default:
    identifier: ko-guides-core-registry-create_registry
    parent: registry
weight: 2
---

사용자 정의 레지스트리는 사용할 수 있는 아티팩트 유형에 대한 유연성과 제어 기능을 제공하고 레지스트리의 가시성을 제한하는 등의 작업을 수행할 수 있도록 합니다.

{{% pageinfo color="info" %}}
코어 및 사용자 정의 레지스트리에 대한 전체 비교는 [레지스트리 유형]({{< relref path="registry_types.md#summary" lang="ko" >}})의 요약 표를 참조하세요.
{{% /pageinfo %}}

## 사용자 정의 레지스트리 만들기

사용자 정의 레지스트리를 만들려면 다음을 수행하세요.

1. https://wandb.ai/registry/의 **Registry** 앱으로 이동합니다.
2. **Custom registry** 내에서 **Create registry** 버튼을 클릭합니다.
3. **Name** 필드에 레지스트리 이름을 입력합니다.
4. 필요에 따라 레지스트리에 대한 설명을 제공합니다.
5. **Registry visibility** 드롭다운에서 레지스트리를 볼 수 있는 사용자를 선택합니다. 레지스트리 visibility 옵션에 대한 자세한 내용은 [레지스트리 visibility types]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ko" >}})을 참조하세요.
6. **Accepted artifacts type** 드롭다운에서 **All types** 또는 **Specify types**를 선택합니다.
7. (**Specify types**를 선택한 경우) 레지스트리가 허용하는 아티팩트 유형을 하나 이상 추가합니다.
8. **Create registry** 버튼을 클릭합니다.

{{% alert %}}
아티팩트 유형은 레지스트리 설정에 저장된 후에는 레지스트리에서 제거할 수 없습니다.
{{% /alert %}}

예를 들어, 다음 이미지는 사용자가 생성하려는 `Fine_Tuned_Models`라는 사용자 정의 레지스트리를 보여줍니다. 레지스트리는 수동으로 레지스트리에 추가된 멤버만 **Restricted**됩니다.

{{< img src="/images/registry/create_registry.gif" alt="" >}}

## Visibility Types

레지스트리의 *visibility*는 해당 레지스트리에 액세스할 수 있는 사용자를 결정합니다. 사용자 정의 레지스트리의 visibility을 제한하면 지정된 멤버만 해당 레지스트리에 액세스할 수 있습니다.

사용자 정의 레지스트리에 대한 두 가지 유형의 레지스트리 가시성 옵션이 있습니다.

| 가시성 | 설명 |
| --- | --- |
| Restricted | 초대된 조직 멤버만 레지스트리에 액세스할 수 있습니다. |
| Organization | 조직의 모든 사용자가 레지스트리에 액세스할 수 있습니다. |

팀 관리자 또는 레지스트리 관리자는 사용자 정의 레지스트리의 가시성을 설정할 수 있습니다.

Restricted 가시성으로 사용자 정의 레지스트리를 생성하는 사용자는 레지스트리 관리자로 레지스트리에 자동으로 추가됩니다.

## 사용자 정의 레지스트리의 visibility 구성

팀 관리자 또는 레지스트리 관리자는 사용자 정의 레지스트리를 생성하는 동안 또는 생성 후에 사용자 정의 레지스트리의 visibility를 할당할 수 있습니다.

기존 사용자 정의 레지스트리의 visibility를 제한하려면 다음을 수행하세요.

1. https://wandb.ai/registry/의 **Registry** 앱으로 이동합니다.
2. 레지스트리를 선택합니다.
3. 오른쪽 상단 모서리에 있는 톱니바퀴 아이콘을 클릭합니다.
4. **Registry visibility** 드롭다운에서 원하는 레지스트리 visibility을 선택합니다.
5. **Restricted visibility**를 선택한 경우:
   1. 이 레지스트리에 액세스할 수 있도록 하려는 조직 멤버를 추가합니다. **Registry members and roles** 섹션으로 스크롤하여 **Add member** 버튼을 클릭합니다.
   2. **Member** 필드 내에서 추가하려는 멤버의 이메일 또는 사용자 이름을 추가합니다.
   3. **Add new member**를 클릭합니다.

{{< img src="/images/registry/change_registry_visibility.gif" alt="" >}}

팀 관리자가 사용자 정의 레지스트리를 만들 때 visibility를 할당하는 방법에 대한 자세한 내용은 [사용자 정의 레지스트리 만들기]({{< relref path="./create_registry.md#create-a-custom-registry" lang="ko" >}})를 참조하세요.
