---
title: Configure registry access
menu:
  default:
    identifier: ko-guides-core-registry-configure_registry
    parent: registry
weight: 3
---

Registry 관리자는 [레지스트리 역할 구성]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}}), [사용자 추가]({{< relref path="configure_registry.md#add-a-user-or-a-team-to-a-registry" lang="ko" >}}), 또는 [사용자 제거]({{< relref path="configure_registry.md#remove-a-user-or-team-from-a-registry" lang="ko" >}})를 레지스트리의 설정을 구성하여 레지스트리에서 수행할 수 있습니다.

## 사용자 관리

### 사용자 또는 팀 추가

Registry 관리자는 개별 user 또는 전체 Teams를 Registry에 추가할 수 있습니다. Registry에 user 또는 Team을 추가하려면 다음을 수행하십시오.

1. Registry(https://wandb.ai/registry/)로 이동합니다.
2. user 또는 Team을 추가할 Registry를 선택합니다.
3. 오른쪽 상단 모서리에 있는 톱니바퀴 아이콘을 클릭하여 Registry settings에 엑세스합니다.
4. **Registry access** 섹션에서 **Add access**를 클릭합니다.
5. **Include users and teams** 필드에 하나 이상의 user 이름, 이메일 또는 Team 이름을 지정합니다.
6. **Add access**를 클릭합니다.

{{< img src="/images/registry/add_team_registry.gif" alt="UI를 사용하여 팀과 개별 사용자를 레지스트리에 추가하는 애니메이션" >}}

[레지스트리에서 user 역할 구성]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}}) 또는 [Registry 역할 권한]({{< relref path="configure_registry.md#registry-role-permissions" lang="ko" >}})에 대해 자세히 알아보세요.

### 사용자 또는 팀 제거
Registry 관리자는 개별 user 또는 전체 Teams를 Registry에서 제거할 수 있습니다. Registry에서 user 또는 Team을 제거하려면 다음을 수행하십시오.

1. Registry(https://wandb.ai/registry/)로 이동합니다.
2. user를 제거할 Registry를 선택합니다.
3. 오른쪽 상단 모서리에 있는 톱니바퀴 아이콘을 클릭하여 Registry settings에 엑세스합니다.
4. **Registry access** 섹션으로 이동하여 제거할 user 이름, 이메일 또는 Team을 입력합니다.
5. **Delete** 버튼을 클릭합니다.

{{% alert %}}
Team에서 user를 제거하면 해당 user의 Registry에 대한 엑세스 권한도 제거됩니다.
{{% /alert %}}

## Registry 역할

Registry의 각 user는 해당 Registry에서 수행할 수 있는 작업을 결정하는 *Registry 역할*을 가집니다.

W&B는 user 또는 Team이 Registry에 추가될 때 자동으로 기본 Registry 역할을 할당합니다.

| Entity | 기본 Registry 역할 |
| ----- | ----- |
| Team | Viewer |
| User (관리자 아님) | Viewer |
| Org 관리자 | Admin |

Registry 관리자는 Registry에서 user 및 Teams에 대한 역할을 할당하거나 수정할 수 있습니다.
자세한 내용은 [레지스트리에서 user 역할 구성]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}})을 참조하십시오.

{{% alert title="W&B 역할 유형" %}}
W&B에는 두 가지 다른 유형의 역할이 있습니다. [Team 역할]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) 및 [Registry 역할]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}}).

Team에서의 역할은 Registry에서의 역할에 영향을 미치거나 관계가 없습니다.
{{% /alert %}}

다음 표는 user가 가질 수 있는 다양한 역할과 해당 권한을 나열합니다.

| 권한 | 권한 그룹 | Viewer | Member | Admin |
|--------------------------------------------------------------- |------------------|--------|--------|-------|
| 컬렉션의 세부 정보 보기 | 읽기 | X | X | X |
| 연결된 아티팩트의 세부 정보 보기 | 읽기 | X | X | X |
| 사용량: use_artifact를 사용하여 레지스트리에서 아티팩트 사용 | 읽기 | X | X | X |
| 연결된 아티팩트 다운로드 | 읽기 | X | X | X |
| 아티팩트의 파일 뷰어에서 파일 다운로드 | 읽기 | X | X | X |
| 레지스트리 검색 | 읽기 | X | X | X |
| 레지스트리의 설정 및 user 목록 보기 | 읽기 | X | X | X |
| 컬렉션에 대한 새 자동화 생성 | 생성 | | X | X |
| 새 버전이 추가될 때 Slack 알림 켜기 | 생성 | | X | X |
| 새 컬렉션 생성 | 생성 | | X | X |
| 새 사용자 지정 레지스트리 생성 | 생성 | | X | X |
| 컬렉션 카드 편집(설명) | 업데이트 | | X | X |
| 연결된 아티팩트 설명 편집 | 업데이트 | | X | X |
| 컬렉션의 태그 추가 또는 삭제 | 업데이트 | | X | X |
| 연결된 아티팩트에서 에일리어스 추가 또는 삭제 | 업데이트 | | X | X |
| 새 아티팩트 연결 | 업데이트 | | X | X |
| 레지스트리에 허용된 유형 목록 편집 | 업데이트 | | X | X |
| 사용자 지정 레지스트리 이름 편집 | 업데이트 | | X | X |
| 컬렉션 삭제 | 삭제 | | X | X |
| 자동화 삭제 | 삭제 | | X | X |
| 레지스트리에서 아티팩트 연결 해제 | 삭제 | | X | X |
| 레지스트리에 대해 허용된 아티팩트 유형 편집 | Admin | | | X |
| 레지스트리 가시성 변경(Organization 또는 Restricted) | Admin | | | X |
| 레지스트리에 user 추가 | Admin | | | X |
| 레지스트리에서 user의 역할 할당 또는 변경 | Admin | | | X |

### 상속된 권한

Registry에서 user의 권한은 개별적으로 또는 Team 멤버십에 의해 해당 user에게 할당된 최고 수준의 권한에 따라 달라집니다.

예를 들어, Registry 관리자가 Nico라는 user를 Registry A에 추가하고 **Viewer** Registry 역할을 할당한다고 가정합니다. 그런 다음 Registry 관리자가 Foundation Model Team이라는 Team을 Registry A에 추가하고 Foundation Model Team에 **Member** Registry 역할을 할당합니다.

Nico는 Registry의 **Member**인 Foundation Model Team의 멤버입니다. **Member**는 **Viewer**보다 더 많은 권한을 가지고 있기 때문에 W&B는 Nico에게 **Member** 역할을 부여합니다.

다음 표는 user의 개별 Registry 역할과 해당 멤버인 Team의 Registry 역할 간의 충돌이 발생할 경우 최고 수준의 권한을 보여줍니다.

| Team Registry 역할 | 개별 Registry 역할 | 상속된 Registry 역할 |
| ------ | ------ | ------ |
| Viewer | Viewer | Viewer |
| Member | Viewer | Member |
| Admin | Viewer | Admin |

충돌이 있는 경우 W&B는 user 이름 옆에 최고 수준의 권한을 표시합니다.

예를 들어, 다음 이미지에서 Alex는 `smle-reg-team-1` Team의 멤버이기 때문에 **Member** 역할 권한을 상속받습니다.

{{< img src="/images/registry/role_conflict.png" alt="사용자가 팀의 일부이기 때문에 멤버 역할을 상속합니다." >}}

## Registry 역할 구성
1. Registry(https://wandb.ai/registry/)로 이동합니다.
2. 구성할 Registry를 선택합니다.
3. 오른쪽 상단 모서리에 있는 톱니바퀴 아이콘을 클릭합니다.
4. **Registry members and roles** 섹션으로 스크롤합니다.
5. **Member** 필드 내에서 권한을 편집할 user 또는 Team을 검색합니다.
6. **Registry role** 열에서 user의 역할을 클릭합니다.
7. 드롭다운에서 user에게 할당할 역할을 선택합니다.
