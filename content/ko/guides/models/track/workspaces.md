---
title: Experiments 결과 보기
description: 인터랙티브 시각화로 run 데이터 를 탐색할 수 있는 플레이그라운드
menu:
  default:
    identifier: ko-guides-models-track-workspaces
    parent: experiments
weight: 4
---

W&B workspace는 차트 커스터마이즈와 모델 결과를 탐색할 수 있는 개인 맞춤형 공간입니다. W&B workspace는 *Tables*와 *Panel sections*로 구성되어 있습니다:

* **Tables**: 프로젝트에 기록된 모든 run이 프로젝트 테이블에 나열됩니다. run을 선택하거나 해제하고, 색상을 변경하거나, 테이블을 확장하여 각각의 run에 대한 노트, config, summary metrics를 확인할 수 있습니다.
* **Panel sections**: 하나 이상의 [panels]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}})로 이루어진 영역입니다. 새로운 패널을 만들고, 정리하거나, 리포트로 내보내서 workspace의 스냅샷을 저장할 수 있습니다.

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="Workspace table and panels" >}}

## Workspace 종류
워크스페이스에는 크게 **Personal workspaces**와 **Saved views** 두 가지가 있습니다.

* **Personal workspaces:** 모델 및 데이터 시각화의 심층 분석을 위한 맞춤형 workspace입니다. workspace의 소유자만 편집하고 변경 사항을 저장할 수 있습니다. 팀원들은 personal workspace를 볼 수 있지만, 다른 사람의 personal workspace는 수정할 수 없습니다.
* **Saved views:** Saved views는 workspace의 협업용 스냅샷입니다. 팀의 누구나 saved workspace view를 보고, 편집하고, 저장할 수 있습니다. Saved workspace views를 사용하면 experiments, runs 등을 리뷰하거나 논의할 수 있습니다.

다음 이미지는 Cécile-parker의 팀원들이 만든 여러 personal workspace를 보여줍니다. 이 프로젝트에는 saved view가 없습니다:
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="No saved views" >}}

## Saved workspace views
팀 협업을 위해 맞춤형 workspace view를 만들어 보세요. Saved Views를 활용하면 선호하는 차트와 데이터를 체계적으로 구성할 수 있습니다.

### 새 saved workspace view 만들기

1. personal workspace나 saved view로 이동합니다.
2. workspace를 수정합니다.
3. 오른쪽 상단의 점 세 개(수직 점 세 개) 메뉴를 클릭하고 **새 뷰로 저장**을 선택합니다.

새 saved view는 workspace의 네비게이션 메뉴에 나타납니다.

{{< img src="/images/app_ui/Menu_Views.jpg" alt="Saved views menu" >}}

### Saved workspace view 업데이트
저장된 변경사항은 saved view의 이전 상태를 덮어씁니다. 저장하지 않은 변경사항은 유지되지 않습니다. W&B에서 saved workspace view를 업데이트하려면:

1. saved view로 이동합니다.
2. workspace 내 차트와 데이터를 원하는 대로 수정합니다.
3. **Save** 버튼을 클릭해서 변경사항을 저장합니다.

{{% alert %}}
변경된 내용을 저장하면 확인 다이얼로그가 나타납니다. 앞으로 이 알림창을 보고 싶지 않다면, **Do not show this modal next time**을 선택한 후 저장을 확정하세요.
{{% /alert %}}

### Saved workspace view 삭제
더 이상 필요 없는 saved view를 삭제할 수 있습니다.

1. 삭제하려는 saved view로 이동합니다.
2. 오른쪽 상단의 점 세 개(**...**)를 선택합니다.
3. **Delete view**를 선택합니다.
4. 삭제를 확정하면 해당 view가 workspace 메뉴에서 제거됩니다.

### Workspace view 공유하기
workspace의 맞춤형 View를 팀과 공유하고 싶다면 해당 workspace의 URL을 직접 공유하세요. workspace project에 엑세스 권한이 있는 모든 사용자는 그 workspace의 saved Views를 볼 수 있습니다.

## Workspace templates
{{% alert %}}이 기능은 [Enterprise](https://wandb.ai/site/pricing/) 라이선스가 필요합니다.{{% /alert %}}

_workspace templates_를 이용해, 기존 workspace와 동일한 설정으로 새 workspace를 신속하게 만들 수 있습니다. 즉, [새 workspace의 기본 설정]({{< relref path="#default-workspace-settings" lang="ko" >}}) 대신 기존 workspace의 설정을 활용할 수 있습니다. 현재 workspace template에서는 [line plot 설정]({{< relref path="/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" lang="ko" >}})을 맞춤화할 수 있습니다.

### 기본 workspace 설정
기본적으로 새 workspace는 line plot에 대해 다음과 같은 설정을 사용합니다:

| 설정 | 기본값 |
|-------|----------
| X축                | Step |
| 스무딩 타입         | Time weight EMA |
| 스무딩 가중치       | 0 |
| 최대 run 수        | 10 |
| 차트 내 그룹화      | 켜짐 |
| 그룹 집계           | 평균(Mean) |

### Workspace template 설정하기
1. 아무 workspace나 열거나 새로 만듭니다.
1. [line plot 설정]({{< relref path="/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" lang="ko" >}})을 원하는 대로 조정하세요.
1. 설정을 workspace template로 저장하세요:
    1. workspace 상단에서 **Undo**, **Redo** 아이콘 근처의 `...` 액션 메뉴를 클릭합니다.
    1. **Save personal workspace template**를 클릭합니다.
    1. template의 line plot 설정을 확인한 뒤, **Save**를 클릭하세요.

이제 새로 만드는 workspace에는 기본값 대신 이 설정이 적용됩니다.

### Workspace template 확인하기
workspace template의 현재 설정을 보려면:
1. 아무 페이지에서나 오른쪽 상단의 사용자 아이콘을 클릭 후 드롭다운 메뉴에서 **Settings**를 선택합니다.
1. **Personal workspace template** 섹션으로 이동합니다. workspace template을 사용 중이면 해당 설정이 표시됩니다. 아니면 상세 정보가 없습니다.

### Workspace template 업데이트하기
workspace template을 업데이트하려면:

1. 아무 workspace나 엽니다.
1. workspace의 설정을 원하는 대로 수정합니다. 예를 들어 run 포함 개수를 `11`로 변경할 수 있습니다.
1. 변경사항을 template에 반영하려면, **Undo**, **Redo** 아이콘 근처의 `...` 액션 메뉴를 클릭한 뒤 **Update personal workspace template**를 클릭하세요.
1. 설정을 확인하고 **Update**를 클릭합니다. template이 업데이트되어, 이 template을 사용하는 모든 workspace에 재적용됩니다.

### Workspace template 삭제하기
workspace template을 삭제하고 기본 설정으로 돌아가려면:

1. 아무 페이지에서나 오른쪽 상단의 사용자 아이콘을 클릭 후 드롭다운에서 **Settings**를 선택합니다.
1. **Personal workspace template** 섹션으로 이동합니다. workspace template의 설정이 표시됩니다.
1. **Settings** 옆의 휴지통 아이콘을 클릭하세요. 

{{% alert %}}
전용 클라우드(Dedicated Cloud) 및 Self-Managed 환경에서는 v0.70 이상에서 workspace template 삭제가 지원됩니다. 이전 버전의 서버에서는 template을 [기본 설정]({{< relref path="#default-workspace-settings" lang="ko" >}})으로 업데이트 하여 사용해 주세요.
{{% /alert %}}

## 프로그램적으로 workspace 만들기

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main)는 [W&B](https://wandb.ai/) workspace와 reports를 프로그램적으로 다룰 수 있게 해주는 Python 라이브러리입니다.

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main)를 쓰면 workspace를 프로그래밍적으로 정의할 수 있습니다. [W&B](https://wandb.ai/) workspaces와 reports를 자동화할 수 있습니다.

workspace의 특성을 다음과 같이 설정할 수 있습니다:

* 패널 레이아웃, 색상, 섹션 순서 지정
* 기본 x축, 섹션 순서, 섹션 접힘 상태 등 workspace 설정 지정
* 각 섹션에 패널을 추가하거나 커스터마이즈해 workspace view 구성
* URL로 기존 workspace 불러와서 수정
* 변경사항을 기존 workspace에 저장하거나 새 view로 저장
* 간단한 식(expression)으로 run을 필터, 그룹화, 정렬
* run의 색상이나 표시 여부 등 외형 지정
* 한 workspace의 view를 다른 workspace로 복사해 연동 및 재사용

### Workspace API 설치

`wandb` 외에도, `wandb-workspaces` 역시 설치해야 합니다:

```bash
pip install wandb wandb-workspaces
```

### 프로그래밍적으로 workspace view 정의 및 저장하기

```python
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### 기존 view 수정하기
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### workspace의 `saved view`를 다른 workspace로 복사하기

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```

workspace API 예시는 [`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces)에서 확인하실 수 있습니다. 처음부터 끝까지 배우고 싶다면 [Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ko" >}}) 튜토리얼을 참고하세요.