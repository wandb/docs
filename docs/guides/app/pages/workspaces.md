---
title: Manage workspaces
description: run 데이터를 인터랙티브한 시각화로 탐색할 수 있는 플레이그라운드
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B workspace는 차트를 사용자 지정하고 모델 결과를 탐색할 수 있는 개인 샌드박스입니다. W&B workspace는 *테이블*과 *패널 섹션*으로 구성됩니다:

* **Tables**: 프로젝트에 로그된 모든 run이 프로젝트의 테이블에 나열됩니다. run을 켜고 끄고, 색상을 변경하며, 각 run에 대한 노트, 설정 및 요약 메트릭을 보려면 테이블을 확장하세요.
* **Panel sections**: 하나 이상의 [패널](../features/panels/intro.md)을 포함하는 섹션입니다. 새 패널을 만들고, 구성하고, 리포트로 내보내어 워크스페이스의 스냅샷을 저장하세요.

![](/images/app_ui/workspace_table_and_panels.png)

## Workspace types
워크스페이스는 두 가지 주요 카테고리로 나눌 수 있습니다: **개인 워크스페이스**와 **저장된 뷰**.

* **개인 워크스페이스:** 모델 및 데이터 시각화의 심층 분석을 위한 사용자 지정 가능한 워크스페이스입니다. 워크스페이스의 소유자만이 편집하고 변경 사항을 저장할 수 있습니다. 팀원은 개인 워크스페이스를 볼 수 있지만 다른 사람의 개인 워크스페이스를 변경할 수는 없습니다.
* **저장된 뷰:** 저장된 뷰는 워크스페이스의 협업 가능한 스냅샷입니다. 팀의 누구라도 저장된 워크스페이스 뷰를 보고, 편집하고, 변경 사항을 저장할 수 있습니다. 저장된 워크스페이스 뷰를 사용하여 실험, run 등을 검토하고 논의하세요.

다음 이미지는 Cécile-parker의 팀원들이 만든 여러 개인 워크스페이스를 보여줍니다. 이 프로젝트에는 저장된 뷰가 없습니다:
![](/images/app_ui/Menu_No_views.jpg)

## Saved workspace views
맞춤형 워크스페이스 뷰로 팀 협업을 개선하세요. 차트와 데이터를 구성하기 위해 저장된 뷰를 만드세요.

### Create a new saved workspace view

1. 개인 워크스페이스 또는 저장된 뷰로 이동합니다.
2. 워크스페이스를 편집합니다.
3. 워크스페이스의 오른쪽 상단 모서리에 있는 3개의 가로 점으로 된 메뉴 아이콘을 클릭합니다. **새 뷰로 저장**을 클릭합니다.

새로운 저장된 뷰는 워크스페이스 탐색 메뉴에 나타납니다.

![](/images/app_ui/Menu_Views.jpg)

### Update a saved workspace view 
저장된 변경 사항은 저장된 뷰의 이전 상태를 덮어씁니다. 저장되지 않은 변경 사항은 유지되지 않습니다. W&B에서 저장된 워크스페이스 뷰를 업데이트하려면:

1. 저장된 뷰로 이동합니다.
2. 워크스페이스 내에서 원하는 차트와 데이터 변경을 수행합니다.
3. **저장** 버튼을 클릭하여 변경 사항을 확인합니다.

:::info
변경 사항을 워크스페이스 뷰에 저장할 때 확인 대화 상자가 나타납니다. 이후에 이 프롬프트를 보지 않으려면 저장을 확인하기 전에 **다음 번에 이 모달 표시 안 함** 옵션을 선택하세요.
:::

### Delete a saved workspace view
더 이상 필요 없는 저장된 뷰를 제거하세요.

1. 제거하려는 저장된 뷰로 이동합니다.
2. 뷰의 오른쪽 상단에 있는 햄버거 메뉴(3개의 가로 줄)를 클릭합니다.
3. **뷰 삭제**를 선택합니다.
4. 삭제를 확인하여 뷰를 워크스페이스 메뉴에서 제거합니다.

![](/images/app_ui/Deleting.gif)

### Share a workspace view
워크스페이스 URL을 직접 공유하여 사용자 지정된 워크스페이스를 팀과 공유하세요. 워크스페이스 프로젝트에 엑세스 권한이 있는 모든 사용자는 해당 워크스페이스의 저장된 뷰를 볼 수 있습니다.

# Programmatic workspace

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main)는 [Weights & Biases](https://wandb.ai/) 워크스페이스 및 리포트를 프로그래밍 방식으로 작업하기 위한 Python 라이브러리입니다.

## Creating a workspace programmatically

워크스페이스의 이름, 연관된 엔터티와 프로젝트 및 포함해야 할 섹션과 같은 속성을 정의하여 프로그래밍 방식으로 워크스페이스를 정의할 수 있습니다.

- **Programmatic workspace creation:**
  - 특정 설정으로 워크스페이스를 정의하고 생성합니다.
  - 패널 레이아웃, 색상 및 섹션 순서를 설정합니다.
- **Workspace customization:**
  - 기본 x축, 섹션 순서 및 축소 상태와 같은 워크스페이스 설정을 구성합니다.
  - 섹션 내에서 패널을 추가하고 사용자 지정하여 워크스페이스 뷰를 구성합니다.
- **Editing existing workspace `saved views`:**
  - URL을 사용하여 기존 워크스페이스를 로드하고 수정합니다.
  - 기존 워크스페이스에 변경 사항을 저장하거나 새 뷰로 저장합니다.
- **Run filtering and grouping:**
  - 간단한 표현을 사용하여 run을 프로그래밍 방식으로 필터링, 그룹화 및 정렬합니다.
  - 색상 및 가시성과 같은 설정으로 run의 모양을 사용자 지정합니다.
- **Cross-workspace integration:**
  - 하나의 워크스페이스에서 다른 워크스페이스로 뷰를 복사하여 원활한 통합 및 재사용을 가능하게 합니다.

프로그래밍 방식으로 워크스페이스를 생성하고 편집하는 방법에 대한 자세한 내용은 [Programmatic Workspaces](../../../tutorials/workspaces.md) 튜토리얼을 참조하세요.

### Install Workspace API

`wandb` 외에도 `wandb-workspaces`를 설치했는지 확인하세요:

```bash
pip install wandb wandb-workspaces
```

## Example Workspace API workflows
다음은 W&B Workspace API를 사용하여 수행할 수 있는 일반적인 작업의 목록입니다.

포괄적인 워크스페이스 API 예제는 [`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces)를 참조하세요. 처음부터 끝까지 튜토리얼은 [Programmatic Workspaces](../../../tutorials/workspaces.md) 튜토리얼을 참조하세요.

### Define and save a workspace view programmatically

```python
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### Edit an existing view
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### Copy a workspace `saved view` to another workspace

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```