---
title: 레지스트리 엑세스 구성
menu:
  default:
    identifier: ko-guides-core-registry-configure_registry
    parent: registry
weight: 3
---

레지스트리 관리자는 레지스트리의 설정에서 [레지스트리 역할 구성]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}}), [사용자 추가]({{< relref path="configure_registry.md#add-a-user-or-a-team-to-a-registry" lang="ko" >}}), 또는 [사용자 삭제]({{< relref path="configure_registry.md#remove-a-user-or-team-from-a-registry" lang="ko" >}})를 할 수 있습니다.

## 사용자 관리

### 사용자 또는 팀 추가

레지스트리 관리자는 개별 사용자 또는 전체 팀을 레지스트리에 추가할 수 있습니다. 레지스트리에 사용자 또는 팀을 추가하려면:

1. https://wandb.ai/registry/ 에서 Registry 페이지로 이동합니다.
2. 사용자를 추가할 레지스트리를 선택합니다.
3. 오른쪽 상단의 톱니바퀴 아이콘을 클릭해 레지스트리 설정에 엑세스합니다.
4. **Registry access** 섹션에서 **Add access**를 클릭합니다.
5. **Include users and teams** 입력란에 하나 이상의 사용자 이름, 이메일 또는 팀 이름을 입력합니다.
6. **Add access**를 클릭합니다.

{{< img src="/images/registry/add_team_registry.gif" alt="Adding teams to registry" >}}

[레지스트리에서 사용자 역할 구성하기]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}}) 또는 [Registry 역할 권한]({{< relref path="configure_registry.md#registry-role-permissions" lang="ko" >}})에 대해 더 알아보세요.

### 사용자 또는 팀 삭제
레지스트리 관리자는 개별 사용자 또는 전체 팀을 레지스트리에서 삭제할 수 있습니다. 사용자 또는 팀을 삭제하려면:

1. https://wandb.ai/registry/ 에서 Registry 페이지로 이동합니다.
2. 사용자를 삭제할 레지스트리를 선택합니다.
3. 오른쪽 상단의 톱니바퀴 아이콘을 클릭해 레지스트리 설정에 엑세스합니다.
4. **Registry access** 섹션에서 삭제하려는 사용자 이름, 이메일 또는 팀 이름을 입력합니다.
5. **Delete** 버튼을 클릭합니다.

{{% alert %}}
팀에서 사용자를 삭제하면, 해당 사용자의 레지스트리 엑세스 권한도 함께 삭제됩니다.
{{% /alert %}}

## Registry 역할

각 사용자는 레지스트리에서 *레지스트리 역할*을 가지며, 이 역할에 따라 해당 레지스트리에서 할 수 있는 작업이 달라집니다.

W&B는 사용자가 레지스트리에 추가될 때 기본 레지스트리 역할을 자동으로 할당합니다.

| 엔티티 | 기본 레지스트리 역할 |
| ----- | ----- |
| Team | Viewer |
| User (비관리자) | Viewer |
| Org admin | Admin |

레지스트리 관리자는 레지스트리 내의 사용자 및 팀에 대해 역할을 할당하거나 수정할 수 있습니다.  
자세한 내용은 [레지스트리에서 사용자 역할 구성하기]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}})를 참고하세요.

{{% alert title="W&B 역할 유형" %}}
W&B에는 두 가지 주요 역할 유형이 있습니다: [Team 역할]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}})과 [Registry 역할]({{< relref path="configure_registry.md#configure-registry-roles" lang="ko" >}})

팀에서의 역할은 레지스트리에서의 역할과는 별개로, 영향을 주지 않습니다.
{{% /alert %}}

아래 표는 사용자가 가질 수 있는 다양한 역할과 각 역할별 권한을 보여줍니다:

| 권한                                                     | 권한 그룹    | Viewer | Member | Admin | 
|----------------------------------------------------------|---------------|--------|--------|-------|
| 컬렉션 상세 정보 보기                                     | 읽기         |   X    |   X    |   X   |
| 연결된 artifact 상세 보기                                 | 읽기         |   X    |   X    |   X   |
| use_artifact로 레지스트리의 artifact 사용                 | 읽기         |   X    |   X    |   X   |
| 연결된 artifact 다운로드                                  | 읽기         |   X    |   X    |   X   |
| artifact 파일 뷰어에서 파일 다운로드                       | 읽기         |   X    |   X    |   X   |
| 레지스트리 검색                                           | 읽기         |   X    |   X    |   X   |
| 레지스트리의 설정 및 사용자 목록 보기                     | 읽기         |   X    |   X    |   X   |
| 컬렉션에 대한 새 자동화 생성                              | 생성         |        |   X    |   X   |
| 새 버전이 추가될 때 Slack 알림 켜기                       | 생성         |        |   X    |   X   |
| 새 컬렉션 생성                                            | 생성         |        |   X    |   X   |
| 새 커스텀 레지스트리 생성                                 | 생성         |        |   X    |   X   |
| 컬렉션 카드(설명) 수정                                    | 수정         |        |   X    |   X   |
| 연결된 artifact 설명 수정                                 | 수정         |        |   X    |   X   |
| 컬렉션 태그 추가/삭제                                     | 수정         |        |   X    |   X   |
| 연결된 artifact의 에일리어스 추가/삭제                    | 수정         |        |   X    |   X   |
| 새로운 artifact 연결                                      | 수정         |        |   X    |   X   |
| 레지스트리 허용 타입 목록 수정                            | 수정         |        |   X    |   X   |
| 커스텀 레지스트리 이름 수정                               | 수정         |        |   X    |   X   |
| 컬렉션 삭제                                               | 삭제         |        |   X    |   X   |
| 자동화 삭제                                               | 삭제         |        |   X    |   X   |
| 레지스트리에서 artifact 연결 해제                         | 삭제         |        |   X    |   X   |
| 레지스트리의 허용 artifact 타입 수정                      | 관리자       |        |        |   X   |
| 레지스트리 공개 범위 변경(Organization 또는 Restricted)    | 관리자       |        |        |   X   |
| 레지스트리에 사용자 추가                                  | 관리자       |        |        |   X   |
| 레지스트리에서 사용자의 역할 배정 또는 변경               | 관리자       |        |        |   X   |


### 역할 상속

사용자의 레지스트리 내 권한은, 개별 혹은 팀 소속을 통해 할당된 권한 중 *가장 높은* 권한에 의존합니다.

예를 들어 Nico라는 사용자가 Registry A에 추가되어 **Viewer** 역할을 받았고, 같은 Registry A에 Foundation Model Team이라는 팀이 **Member** 역할로 추가되었습니다.

Nico가 Foundation Model Team의 멤버일 경우, 이 팀이 Registry에서 **Member** 역할이 되기 때문에 **Member**가 **Viewer**보다 높은 권한을 갖게 되고, W&B는 Nico에게 **Member** 권한을 부여합니다.

아래 표는 사용자의 개별 역할과 팀 역할이 충돌할 때 상속되는 가장 높은 권한을 보여줍니다:

| Team registry role | Individual registry role | Inherited registry role |
| ------ | ------ | ------ | 
| Viewer | Viewer | Viewer |
| Member | Viewer | Member |
| Admin  | Viewer | Admin  | 

권한 충돌이 발생할 경우, W&B는 사용자 이름 옆에 *가장 높은 권한*을 표시합니다.

예시 이미지에서, Alex는 `smle-reg-team-1` 팀의 멤버이기 때문에 **Member** 역할을 상속받아 권한을 부여받습니다.

{{< img src="/images/registry/role_conflict.png" alt="Registry role conflict resolution" >}}


## Registry 역할 구성하기
1. https://wandb.ai/registry/ 에서 Registry 페이지로 이동합니다.
2. 설정할 레지스트리를 선택합니다.
3. 오른쪽 상단의 톱니바퀴(gear) 아이콘을 클릭합니다.
4. **Registry members and roles** 섹션까지 스크롤을 내립니다.
5. **Member** 필드에서 권한을 수정하고 싶은 사용자나 팀을 검색합니다.
6. **Registry role** 컬럼에서 해당 사용자의 역할을 클릭합니다.
7. 드롭다운에서 해당 사용자에게 할당할 역할을 선택합니다.
