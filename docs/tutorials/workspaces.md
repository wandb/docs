---
title: Programmatic Workspaces
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb'/>

`wandb-workspaces` 라이브러리를 사용하여 설정을 정의하고, 패널 레이아웃을 설정하고, 섹션을 구성하여 프로그래밍 방식으로 워크스페이스를 생성, 관리, 사용자 지정하여 기계학습 실험을 더 효과적으로 조직하고 시각화하세요. URL로 워크스페이스를 불러오고 수정하며, 표현식을 사용하여 run을 필터하고 그룹화하며, run 외관을 사용자 지정할 수 있습니다.

`wandb-workspaces`는 Python 라이브러리로, W&B [Workspaces](/guides/app/pages/workspaces) 및 [Reports](/guides/reports)를 프로그래밍 방식으로 생성하고 사용자 지정하는 데 사용됩니다.

이 튜토리얼에서는 `wandb-workspaces`를 사용하여 설정을 정의하고, 패널 레이아웃을 설정하고, 섹션을 구성하여 워크스페이스를 생성하고 사용자 지정하는 방법을 배웁니다.

### 이 노트북을 사용하는 방법
* 각 셀을 한 번에 하나씩 실행하세요.
* 셀을 실행한 후 출력된 URL을 복사하여 워크스페이스에 적용된 변경 사항을 확인하세요.

:::info
워크스페이스와의 프로그래밍 방식의 상호작용은 현재 [**Saved workspaces views**](/guides/app/pages/workspaces#saved-workspace-views)를 지원합니다. Saved workspaces views는 워크스페이스의 공동 작업 가능한 스냅샷입니다. 팀의 누구나 saved workspace views를 보고, 편집하고, 변경 사항을 저장할 수 있습니다.
:::

## 1. 디펜던시 설치 및 가져오기

```python
# 디펜던시 설치
!pip install wandb wandb-workspaces rich
```

```python
# 디펜던시 가져오기
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # 패널 추가를 위해 Reports API 사용

# 출력 형식 개선
%load_ext rich
```

## 2. 새 프로젝트 및 워크스페이스 생성

이 튜토리얼에서는 `wandb_workspaces` API를 실험하기 위해 새 프로젝트를 생성합니다.

참고: 고유한 `Saved view` URL을 사용하여 기존 워크스페이스를 불러올 수 있습니다. 다음 코드 블록에서 이를 수행하는 방법을 확인하세요.

```python
# Weights & Biases 초기화 및 로그인
wandb.login()

# 새로운 프로젝트를 생성하고 샘플 데이터를 로그하는 함수
def create_project_and_log_data():
    project = "workspace-api-example"  # 기본 프로젝트 이름

    # 샘플 데이터를 로그하기 위해 run을 초기화
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

# 새 프로젝트 생성 및 데이터 로그
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### (옵션) 기존 프로젝트 및 워크스페이스 불러오기
새 프로젝트를 생성하는 대신, 기존 프로젝트와 워크스페이스를 불러올 수 있습니다. 이를 위해 고유한 워크스페이스 URL을 찾아 `ws.Workspace.from_url`에 문자열로 전달하세요. URL 형식은 `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc`입니다.

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

## 3. 프로그래밍 방식의 워크스페이스 예제
아래는 프로그래밍 방식의 워크스페이스 기능을 사용하는 예제들입니다.

```python
# 워크스페이스, 섹션, 패널에 대해 사용 가능한 모든 설정 보기
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view`가 있는 워크스페이스 생성
이 예제는 워크스페이스를 생성하고 섹션과 패널로 채우는 방법을 보여줍니다. 워크스페이스는 일반 Python 오브젝트처럼 편집할 수 있어 유연성과 사용 편의성을 제공합니다.

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

### URL에서 워크스페이스 불러오기
원본 설정에 영향을 주지 않고 워크스페이스를 복제하고 사용자 지정할 수 있습니다. 이를 위해 기존 워크스페이스를 불러와 새 뷰로 저장하세요.

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

이제 워크스페이스의 이름이 "Updated Workspace Name"으로 변경된 것을 확인하세요.

### 기본 설정
다음 코드는 워크스페이스를 생성하고 패널이 있는 섹션을 추가하며 워크스페이스, 개별 섹션 및 패널에 대한 설정을 구성하는 방법을 보여줍니다.

```python
# 사용자 지정 설정으로 워크스페이스를 생성하고 구성하는 함수
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

# 워크스페이스를 생성하고 구성하기 위해 함수 실행
custom_settings_example(entity, project)
```

이제 "An example workspace"라는 다른 저장된 뷰를 보고 있는 것을 확인하세요.

## Run 사용자 지정
다음 코드 셀은 run을 필터링하고, 색상을 변경하고, 그룹화하고, 정렬하는 방법을 보여줍니다.

각 예제에서 일반적인 워크플로우는 `ws.RunsetSettings`의 적절한 파라미터에 인수로 원하는 사용자 지정을 지정하는 것입니다.

### Run 필터링
Python 표현식과 `wandb.log`로 로그된 메트릭이나 **Created Timestamp**와 같이 run의 일부로 자동 로그된 메트릭으로 필터를 생성할 수 있습니다. 또한 W&B 앱 UI에 나타나는 **Name**, **Tags**, **ID**와 같은 필터를 참조할 수 있습니다.

다음 예제는 검증 손실 요약, 검증 정확도 요약 및 지정된 정규식에 따라 run을 필터링하는 방법을 보여줍니다.

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # 프로젝트의 모든 run 가져오기
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 여러 필터 적용: val_loss < 0.1, val_accuracy > 0.8, 및 run 이름이 regex 패턴과 일치
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' 요약으로 run 필터링
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' 요약으로 run 필터링
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # 's'로 시작하는 run 이름을 일치시키기 위한 regex 검색 추가
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("워크스페이스가 고급 필터 및 정규식 검색과 함께 저장되었습니다.")

advanced_filter_example(entity, project)
```

필터 표현식 리스트를 전달하면 부울 "AND" 논리가 적용된다는 것을 유의하세요.

### Run 색상 변경
이 예제는 워크스페이스에서 run의 색상을 변경하는 방법을 보여줍니다.

```python
def run_color_example(entity: str, project: str) -> None:
    # 프로젝트의 모든 run 가져오기
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run에 색상 동적으로 할당
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
    print("워크스페이스가 run 색상과 함께 저장되었습니다.")

run_color_example(entity, project)
```

### Run 그룹화
이 예제는 특정 메트릭을 기준으로 run을 그룹화하는 방법을 보여줍니다.

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
    print("워크스페이스가 그룹화된 run과 함께 저장되었습니다.")

grouping_example(entity, project)
```

### Run 정렬
이 예제는 검증 손실 요약을 기준으로 run을 정렬하는 방법을 보여줍니다.

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
            order=[ws.Ordering(ws.Summary("val_loss"))] # val_loss 요약을 사용하여 정렬
        )
    )
    workspace.save()
    print("워크스페이스가 정렬된 run과 함께 저장되었습니다.")

sorting_example(entity, project)
```

## 4. 모든 걸 합쳐보기: 포괄적인 예제

이 예제는 포괄적인 워크스페이스를 생성, 설정을 구성하고, 섹션에 패널을 추가하는 방법을 보여줍니다.

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # 프로젝트의 모든 run 가져오기
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run에 색상을 동적으로 할당하고 run 설정 생성
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
    print("워크스페이스가 생성되고 저장되었습니다.")

full_end_to_end_example(entity, project)
```