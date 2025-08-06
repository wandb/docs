---
title: 프로그래밍 방식 워크스페이스
menu:
  tutorials:
    identifier: ko-tutorials-workspaces
    parent: null
weight: 5
---

{{% alert %}}
W&B Report 및 Workspace API는 퍼블릭 프리뷰 단계입니다.
{{% /alert %}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
기계학습 실험을 더 효과적으로 정리하고 시각화하세요. 프로그래밍 방식으로 워크스페이스를 생성, 관리, 커스터마이즈할 수 있습니다. 설정을 정의하고, 패널 레이아웃을 지정하며, 섹션을 체계적으로 구성할 수 있습니다. [`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B 라이브러리를 활용하여 워크스페이스를 관리하세요. 워크스페이스를 URL로 불러오고 수정하거나, 식(expression)으로 run을 필터 및 그룹화하거나, run의 표시 방식을 쉽게 커스터마이즈할 수 있습니다.

`wandb-workspaces`는 프로그래밍 방식으로 W&B [Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ko" >}})와 [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})를 생성하고 커스터마이즈할 수 있는 Python 라이브러리입니다.

이 튜토리얼에서는 `wandb-workspaces`를 사용하여 설정을 정의하고, 패널 레이아웃을 지정하며, 섹션을 구성하여 워크스페이스를 생성 및 커스터마이즈하는 방법을 안내합니다.

## 이 노트북 사용 방법
* 각 셀을 한 번씩 실행하세요.
* 셀 실행 후 출력된 URL을 복사하여 워크스페이스의 변경사항을 확인하세요.

{{% alert %}}
워크스페이스와의 프로그래밍 방식 연동은 현재 [저장된 워크스페이스 뷰]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ko" >}})에서만 지원됩니다. 저장된 워크스페이스 뷰는 워크스페이스의 협업 스냅샷입니다. 팀 내 모든 사용자가 이 뷰를 보고, 수정하고, 변경 사항을 저장할 수 있습니다.
{{% /alert %}}

## 1. 라이브러리 설치 및 임포트

```python
# 패키지 설치
!pip install wandb wandb-workspaces rich
```

```python
# 라이브러리 임포트
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # 패널을 추가할 때 Reports API 사용

# 출력 포맷 향상
%load_ext rich
```

## 2. 새 프로젝트와 워크스페이스 생성

이 튜토리얼에서는 `wandb_workspaces` API를 실험할 수 있도록 새로운 프로젝트를 생성합니다.

참고: 기존 워크스페이스는 고유한 `Saved view` URL을 사용해 불러올 수 있습니다. 다음 코드블록을 참고하세요.

```python
# W&B 초기화 및 로그인
wandb.login()

# 새 프로젝트를 생성하고 샘플 데이터를 로그하는 함수
def create_project_and_log_data():
    project = "workspace-api-example"  # 기본 프로젝트 이름

    # 샘플 데이터를 로그하기 위한 run 초기화
    with wandb.init(project=project, name="sample_run") as run:
        for step in range(100):
            run.log({
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

### (선택 사항) 기존 프로젝트와 워크스페이스 불러오기

새 프로젝트를 생성하는 대신, 본인의 기존 프로젝트와 워크스페이스를 불러올 수 있습니다. 고유한 워크스페이스 URL을 찾아 문자열로 `ws.Workspace.from_url`에 전달하면 됩니다. URL 형태는 `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc` 입니다.

예시:

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc").

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. 워크스페이스 프로그래밍 예시
아래는 워크스페이스 기능을 프로그래밍 방식으로 활용하는 예시입니다:

```python
# 워크스페이스, 섹션, 패널에 사용할 수 있는 모든 설정 보기
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view`로 워크스페이스 생성하기
이 예시에서는 새로운 워크스페이스를 생성하고, 섹션과 패널로 채우는 방법을 보여줍니다. 워크스페이스는 일반 Python 오브젝트처럼 쉽게 수정할 수 있습니다.

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

### URL로 워크스페이스 불러오기
원본 세팅에 영향을 주지 않고 워크스페이스를 복제 및 커스터마이즈할 수 있습니다. 기존 워크스페이스를 불러와서 새로운 뷰로 저장하세요:

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save_as_new_view()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

워크스페이스 이름이 이제 "Updated Workspace Name"으로 변경된 것을 확인할 수 있습니다.

### 기본 설정 적용하기
아래 코드는 워크스페이스를 생성하고 섹션/패널 추가 및 워크스페이스, 각 섹션, 패널별로 설정을 커스터마이즈하는 예시입니다:

```python
# 사용자 정의 설정으로 워크스페이스 생성 및 설정하는 함수
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

# 워크스페이스 생성 및 설정 함수 실행
custom_settings_example(entity, project)
```

이제 "An example workspace"라는 별도의 saved view를 확인할 수 있습니다.

## run 커스터마이즈하기

다음 코드는 run을 필터링, 색상 변경, 그룹화, 정렬하는 방법을 보여줍니다.

각 예시에서 원하는 커스터마이즈를 `ws.RunsetSettings`의 인수로 지정하면 됩니다.

### run 필터링

Python 식과 `wandb.log`로 기록한 메트릭 또는 **Created Timestamp**와 같은 자동 기록 메트릭을 사용하여 run을 필터링할 수 있습니다. W&B App UI에서 보이는 **Name**, **Tags**, **ID**와 같은 필터명도 참조 가능합니다.

다음은 검증 손실, 검증 정확도, 그리고 정규표현식에 기반하여 run을 필터링한 예시입니다:

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # 프로젝트 내 모든 run 가져오기
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 여러 필터 적용: val_loss < 0.1, val_accuracy > 0.8, run name이 정규표현식 패턴과 일치
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' summary로 run 필터
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' summary로 run 필터
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # run 이름이 's'로 시작하는 것만 정규표현식 검색에 추가
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

필터 식을 리스트로 전달하면 논리적 "AND"가 적용됩니다.

### run 색상 변경

아래는 워크스페이스 내 run 색상을 변경하는 예시입니다:

```python
def run_color_example(entity: str, project: str) -> None:
    # 프로젝트 내 모든 run 가져오기
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run에 동적으로 색상 할당
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

### run 그룹화

특정 메트릭을 기준으로 run을 그룹화하는 방법입니다.

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

### run 정렬

아래는 검증 손실 summary 기준으로 run을 정렬하는 예시입니다:

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
            order=[ws.Ordering(ws.Summary("val_loss"))] # val_loss summary 기준 정렬
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. 모두 활용하기: 종합 예제

아래는 종합적으로 워크스페이스를 만들고, 설정을 구성하고, 섹션별로 패널을 추가하는 전체 예제입니다:

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # 프로젝트 내 모든 run 가져오기
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run에 동적으로 색상을 입히고 run settings 생성
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