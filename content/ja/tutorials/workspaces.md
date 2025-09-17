---
title: プログラムによる Workspaces
menu:
  tutorials:
    identifier: ja-tutorials-workspaces
    parent: null
weight: 5
---

{{% alert %}}
W&B Report と Workspace API は Public Preview 中です。
{{% /alert %}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
`wandb-workspaces` を使って Workspace をプログラムで作成・管理・カスタマイズすることで、機械学習 実験をより効果的に整理・可視化できます。設定 を定義し、パネルのレイアウトを設定し、セクションを整理できます。URL から Workspace を読み込んで変更したり、式で run をフィルタ・グループ化したり、run の見た目をカスタマイズしたりできます。

`wandb-workspaces` は、W&B の [Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}}) と [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) をプログラムで作成・カスタマイズするための Python ライブラリです。

このチュートリアルでは、`wandb-workspaces` を使って Workspace を作成・カスタマイズし、設定 を定義し、パネルのレイアウトを設定し、セクションを整理する方法を紹介します。
{{< /cta-button >}}

## How to use this notebook
* 各セルを 1 回ずつ実行してください。 
* セルを実行した後に表示される URL をコピーして貼り付け、Workspace に対する変更を確認してください。


{{% alert %}}
Workspace とのプログラム的なやり取りは、現在 [Saved workspaces views]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ja" >}}) に対してサポートされています。Saved workspace views は Workspace の共同編集用スナップショットです。チーム内の誰でも、Saved workspace views を表示・編集し、変更を保存できます。 
{{% /alert %}}

## 1. Install and import dependencies


```python
# 依存関係をインストール
!pip install wandb wandb-workspaces rich
```


```python
# 依存関係をインポート
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # パネルを追加するために Reports API を使用します

# 出力の書式を改善
%load_ext rich
```

## 2. Create a new project and workspace

このチュートリアルでは、`wandb_workspaces` API を試すために新しい Project を作成します。

Note: 固有の `Saved view` の URL を使えば、既存の Workspace を読み込むこともできます。やり方は次のコードブロックを参照してください。


```python
# W&B を初期化してログイン
wandb.login()

# 新しい Project を作成し、サンプルデータをログする関数
def create_project_and_log_data():
    project = "workspace-api-example"  # 既定の Project 名

    # サンプルデータをログするための run を初期化
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

# 新しい Project を作成してデータをログ
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### (Optional) Load an existing project and workspace
新しい Project を作成する代わりに、あなたの既存の Project と Workspace を読み込むこともできます。そのためには、一意の Workspace の URL を見つけ、それを文字列として `ws.Workspace.from_url` に渡します。URL の形式は `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc` です。

例えば、次のようにします:

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc").

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. Programmatic workspace examples
以下は、Workspace をプログラムで操作する機能の例です。


```python
# Workspace、セクション、パネルで利用可能なすべての設定を確認
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### Create a workspace with `saved view`
この例では、新しい Workspace を作成し、セクションとパネルで埋める方法を示します。Workspace は通常の Python オブジェクトのように編集できるため、柔軟で使いやすいです。


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

### Load a workspace from a URL
元の設定に影響を与えずに Workspace を複製してカスタマイズできます。そのためには、既存の Workspace を読み込み、新しい view として保存します:


```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save_as_new_view()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

Workspace の名前が "Updated Workspace Name" に変わっています。

### Basic settings
以下のコードは、Workspace を作成してパネル付きのセクションを追加し、Workspace・各セクション・各パネルの設定 を構成する方法を示します。


```python
# カスタム設定の Workspace を作成・設定する関数
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

# Workspace を作成・設定する関数を実行
custom_settings_example(entity, project)
```

現在表示しているのは "An example workspace" という別の saved view です。

## Customize runs
以下のコードセルでは、run をプログラムでフィルタ、色変更、グループ化、ソートする方法を示します。 

各例での一般的なワークフローは、`ws.RunsetSettings` の該当パラメータに、望むカスタマイズ内容を引数として指定することです。

### Filter runs
`wandb.log` で記録したメトリクスや、run の一部として自動で記録される **Created Timestamp** などに対して、Python の式でフィルタを作成できます。W&B の App UI に表示される **Name**、**Tags**、**ID** のように参照することもできます。

次の例では、検証損失のサマリー、検証精度 のサマリー、指定した正規表現に基づいて run をフィルタする方法を示します:


```python
def advanced_filter_example(entity: str, project: str) -> None:
    # Project 内のすべての run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 複数のフィルタを適用: val_loss < 0.1、val_accuracy > 0.8、run 名が正規表現に一致
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' のサマリーで run をフィルタ
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' のサマリーで run をフィルタ
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # 's' で始まる run 名に一致するよう正規表現検索を追加
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

フィルタ式のリストを渡すと、ブールの "AND" ロジックが適用されます。

### Change the colors of runs
この例では、Workspace 内の run の色を変更する方法を示します:


```python
def run_color_example(entity: str, project: str) -> None:
    # Project 内のすべての run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run に動的に色を割り当てる
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

### Group runs

この例では、特定のメトリクスで run をグループ化する方法を示します。



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

### Sort runs
この例では、検証損失 のサマリーに基づいて run をソートする方法を示します:


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
            order=[ws.Ordering(ws.Summary("val_loss"))] # val_loss のサマリーで並べ替え
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. Putting it all together: comprehensive example

この例では、包括的な Workspace を作成し、その設定 を構成し、セクションにパネルを追加する方法を示します:


```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # Project 内のすべての run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run に動的に色を割り当て、run 設定を作成
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