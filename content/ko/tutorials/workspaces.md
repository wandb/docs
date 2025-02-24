---
title: Programmatic Workspaces
menu:
  tutorials:
    identifier: ko-tutorials-workspaces
    parent: null
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
프로그래밍 방식으로 워크스페이스를 생성, 관리 및 사용자 정의하여 기계 학습 Experiments를 보다 효과적으로 구성하고 시각화하세요. [`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B 라이브러리를 사용하여 설정을 정의하고, 패널 레이아웃을 설정하고, 섹션을 구성할 수 있습니다. URL로 워크스페이스를 로드 및 수정하고, 표현식을 사용하여 Runs를 필터링 및 그룹화하고, Runs의 모양을 사용자 정의할 수 있습니다.

`wandb-workspaces`는 프로그래밍 방식으로 W&B [Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ko" >}}) 및 [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})를 생성하고 사용자 정의하기 위한 Python 라이브러리입니다.

이 튜토리얼에서는 `wandb-workspaces`를 사용하여 설정을 정의하고, 패널 레이아웃을 설정하고, 섹션을 구성하여 워크스페이스를 생성하고 사용자 정의하는 방법을 보여줍니다.

## 이 노트북 사용 방법
* 각 셀을 한 번에 하나씩 실행합니다.
* 셀을 실행한 후 출력되는 URL을 복사하여 붙여넣어 워크스페이스에 적용된 변경 사항을 확인합니다.

{{% alert %}}
워크스페이스와의 프로그래밍 방식 상호 작용은 현재 [**저장된 워크스페이스 보기**]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ko" >}})에서 지원됩니다. 저장된 워크스페이스 보기는 워크스페이스의 협업 스냅샷입니다. 팀의 모든 구성원이 저장된 워크스페이스 보기를 보고, 편집하고, 변경 사항을 저장할 수 있습니다.
{{% /alert %}}

## 1. 종속성 설치 및 가져오기

```python
# Install dependencies
!pip install wandb wandb-workspaces rich
```

```python
# Import dependencies
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # We use the Reports API for adding panels

# Improve output formatting
%load_ext rich
```

## 2. 새 프로젝트 및 워크스페이스 생성

이 튜토리얼에서는 `wandb_workspaces` API를 사용하여 실험할 수 있도록 새 프로젝트를 만듭니다.

참고: 고유한 `Saved view` URL을 사용하여 기존 워크스페이스를 로드할 수 있습니다. 이 작업을 수행하는 방법은 다음 코드 블록을 참조하세요.

```python
# Initialize Weights & Biases and Login
wandb.login()

# Function to create a new project and log sample data
def create_project_and_log_data():
    project = "workspace-api-example"  # Default project name

    # Initialize a run to log some sample data
    with wandb.init(project=project, name="sample_run") as run:
        for step in range(100):
            wandb.log({
                "Step": step,
                "val_loss": 1.0 / (step + 1),
                "val_accuracy": step / 100.0,
                "train_loss": 1.0 / (step + 2),
                "train_accuracy": step / 110.0,
                "f1_score": step / 100.0,
                "recall": step / 120.0,
            })
    return project

# Create a new project and log data
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### (선택 사항) 기존 프로젝트 및 워크스페이스 로드
새 프로젝트를 만드는 대신 기존 프로젝트 및 워크스페이스를 로드할 수 있습니다. 이렇게 하려면 고유한 워크스페이스 URL을 찾아 문자열로 `ws.Workspace.from_url`에 전달합니다. URL 형식은 `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc`입니다.

예를 들어:

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc").

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. 프로그래밍 방식 워크스페이스 예제
다음은 프로그래밍 방식 워크스페이스 기능 사용 예제입니다.

```python
# See all available settings for workspaces, sections, and panels.
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view`로 워크스페이스 생성
이 예제에서는 새 워크스페이스를 만들고 섹션 및 패널로 채우는 방법을 보여줍니다. 워크스페이스는 일반 Python 오브젝트처럼 편집할 수 있어 유연성과 사용 편의성을 제공합니다.

```python
def sample_workspace_saved_example(entity: str, project: str) -> str:
    workspace: ws.Workspace = ws.Workspace(
        name="Example W&B Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Validation Metrics",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.BarPlot(metrics=["val_accuracy"]),
                    wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
                ],
                is_open=True,
            ),
        ],
    )
    workspace.save()
    print("Sample Workspace saved.")
    return workspace.url

workspace_url: str = sample_workspace_saved_example(entity, project)
```

### URL에서 워크스페이스 로드
원래 설정에 영향을 주지 않고 워크스페이스를 복제하고 사용자 정의합니다. 이렇게 하려면 기존 워크스페이스를 로드하고 새 보기로 저장합니다.

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

이제 워크스페이스 이름이 "Updated Workspace Name"으로 변경되었습니다.

### 기본 설정
다음 코드는 워크스페이스를 만들고, 패널이 있는 섹션을 추가하고, 워크스페이스, 개별 섹션 및 패널에 대한 설정을 구성하는 방법을 보여줍니다.

```python
# Function to create and configure a workspace with custom settings
def custom_settings_example(entity: str, project: str) -> None:
    workspace: ws.Workspace = ws.Workspace(name="An example workspace", entity=entity, project=project)
    workspace.sections = [
        ws.Section(
            name="Validation",
            panels=[
                wr.LinePlot(x="Step", y=["val_loss"]),
                wr.LinePlot(x="Step", y=["val_accuracy"]),
                wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
                wr.ScalarChart(metric="recall", groupby_aggfunc="mean"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="Training",
            panels=[
                wr.LinePlot(x="Step", y=["train_loss"]),
                wr.LinePlot(x="Step", y=["train_accuracy"]),
            ],
            is_open=False,
        ),
    ]

    workspace.settings = ws.WorkspaceSettings(
        x_axis="Step",
        x_min=0,
        x_max=75,
        smoothing_type="gaussian",
        smoothing_weight=20.0,
        ignore_outliers=False,
        remove_legends_from_panels=False,
        tooltip_number_of_runs="default",
        tooltip_color_run_names=True,
        max_runs=20,
        point_visualization_method="bucketing",
        auto_expand_panel_search_results=False,
    )

    section = workspace.sections[0]
    section.panel_settings = ws.SectionPanelSettings(
        x_min=25,
        x_max=50,
        smoothing_type="none",
    )

    panel = section.panels[0]
    panel.title = "Validation Loss Custom Title"
    panel.title_x = "Custom x-axis title"

    workspace.save()
    print("Workspace with custom settings saved.")

# Run the function to create and configure the workspace
custom_settings_example(entity, project)
```

이제 "An example workspace"라는 다른 저장된 보기를 보고 있습니다.

## Runs 사용자 정의
다음 코드 셀은 프로그래밍 방식으로 Runs를 필터링, 색상 변경, 그룹화 및 정렬하는 방법을 보여줍니다.

각 예제에서 일반적인 워크플로우는 `ws.RunsetSettings`에서 해당 파라미터에 대한 인수로 원하는 사용자 정의를 지정하는 것입니다.

### Runs 필터링
Python 표현식과 `wandb.log`로 기록하거나 **생성 타임스탬프**와 같이 Run의 일부로 자동으로 기록되는 메트릭으로 필터를 만들 수 있습니다. **이름**, **태그** 또는 **ID**와 같이 W&B 앱 UI에 나타나는 방식으로 필터를 참조할 수도 있습니다.

다음 예제에서는 검증 손실 요약, 검증 정확도 요약 및 지정된 정규식을 기반으로 Runs를 필터링하는 방법을 보여줍니다.

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # Get all runs in the project
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Apply multiple filters: val_loss < 0.1, val_accuracy > 0.8, and run name matches regex pattern
    workspace: ws.Workspace = ws.Workspace(
        name="Advanced Filtered Workspace with Regex",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Advanced Filtered Section",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            filters=[
                (ws.Summary("val_loss") < 0.1),  # Filter runs by the 'val_loss' summary
                (ws.Summary("val_accuracy") > 0.8),  # Filter runs by the 'val_accuracy' summary
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # Add regex search to match run names starting with 's'
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

필터 표현식 목록을 전달하면 부울 "AND" 논리가 적용됩니다.

### Runs의 색상 변경
이 예제에서는 워크스페이스에서 Runs의 색상을 변경하는 방법을 보여줍니다.

```python
def run_color_example(entity: str, project: str) -> None:
    # Get all runs in the project
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Dynamically assign colors to the runs
    run_colors: list = ['purple', 'orange', 'teal', 'magenta']
    run_settings: dict = {}
    for i, run in enumerate(runs):
        run_settings[run.id] = ws.RunSettings(color=run_colors[i % len(run_colors)])

    workspace: ws.Workspace = ws.Workspace(
        name="Run Colors Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Run Colors Section",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            run_settings=run_settings
        )
    )

    workspace.save()
    print("Workspace with run colors saved.")

run_color_example(entity, project)
```

### Runs 그룹화

이 예제에서는 특정 메트릭으로 Runs를 그룹화하는 방법을 보여줍니다.

```python
def grouping_example(entity: str, project: str) -> None:
    workspace: ws.Workspace = ws.Workspace(
        name="Grouped Runs Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Grouped Runs",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            groupby=[ws.Metric("Name")]
        )
    )
    workspace.save()
    print("Workspace with grouped runs saved.")

grouping_example(entity, project)
```

### Runs 정렬
이 예제에서는 검증 손실 요약을 기준으로 Runs를 정렬하는 방법을 보여줍니다.

```python
def sorting_example(entity: str, project: str) -> None:
    workspace: ws.Workspace = ws.Workspace(
        name="Sorted Runs Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Sorted Runs",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            order=[ws.Ordering(ws.Summary("val_loss"))] #Order using val_loss summary
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. 모두 함께 사용: 포괄적인 예제

이 예제에서는 포괄적인 워크스페이스를 만들고, 설정을 구성하고, 섹션에 패널을 추가하는 방법을 보여줍니다.

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # Get all runs in the project
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Dynamically assign colors to the runs and create run settings
    run_colors: list = ['red', 'blue', 'green', 'orange', 'purple', 'teal', 'magenta', '#FAC13C']
    run_settings: dict = {}
    for i, run in enumerate(runs):
        run_settings[run.id] = ws.RunSettings(color=run_colors[i % len(run_colors)], disabled=False)

    workspace: ws.Workspace = ws.Workspace(
        name="My Workspace Template",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Main Metrics",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                    wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
                ],
                is_open=True,
            ),
            ws.Section(
                name="Additional Metrics",
                panels=[
                    wr.ScalarChart(metric="precision", groupby_aggfunc="mean"),
                    wr.ScalarChart(metric="recall", groupby_aggfunc="mean"),
                ],
            ),
        ],
        settings=ws.WorkspaceSettings(
            x_axis="Step",
            x_min=0,
            x_max=100,
            smoothing_type="none",
            smoothing_weight=0,
            ignore_outliers=False,
            remove_legends_from_panels=False,
            tooltip_number_of_runs="default",
            tooltip_color_run_names=True,
            max_runs=20,
            point_visualization_method="bucketing",
            auto_expand_panel_search_results=False,
        ),
        runset_settings=ws.RunsetSettings(
            query="",
            regex_query=False,
            filters=[
                ws.Summary("val_loss") < 1,
                ws.Metric("Name") == "sample_run",
            ],
            groupby=[ws.Metric("Name")],
            order=[ws.Ordering(ws.Summary("Step"), ascending=True)],
            run_settings=run_settings
        )
    )
    workspace.save()
    print("Workspace created and saved.")

full_end_to_end_example(entity, project)
```