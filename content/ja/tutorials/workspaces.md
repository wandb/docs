---
title: Programmatic Workspaces
menu:
  tutorials:
    identifier: ja-tutorials-workspaces
    parent: null
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
[Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})をプログラムで作成、管理、カスタマイズすることで、機械学習の実験をより効果的に整理し、可視化できます。設定を定義し、パネルレイアウトを設定し、[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B ライブラリでセクションを整理できます。URLで[Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})をロードおよび変更したり、式を使用して[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をフィルタリングおよびグループ化したり、[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})の外観をカスタマイズしたりできます。

`wandb-workspaces`は、W&Bの[Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})と[Reports]({{< relref path="/guides/core/reports/" lang="ja" >}})をプログラムで作成およびカスタマイズするためのPythonライブラリです。

このチュートリアルでは、`wandb-workspaces`を使用して、設定を定義し、パネルレイアウトを設定し、セクションを整理することで、[Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})を作成およびカスタマイズする方法を説明します。

## ノートブックの使い方
* 各セルを一度に1つずつ実行します。
* セルを実行した後に表示されるURLをコピーして貼り付け、[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})に加えられた変更を表示します。

{{% alert %}}
[Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})とのプログラムによるインタラクションは、現在[**保存された[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})ビュー**]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ja" >}})でサポートされています。保存された[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})ビューは、[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})の共同スナップショットです。チームの誰でも、保存された[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})ビューを表示、編集、および変更を保存できます。
{{% /alert %}}

## 1. 依存関係のインストールとインポート

```python
# 依存関係をインストールする
!pip install wandb wandb-workspaces rich
```

```python
# 依存関係をインポートする
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # パネルを追加するためにReports APIを使用します

# 出力形式を改善する
%load_ext rich
```

## 2. 新しい[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}})と[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})を作成する

このチュートリアルでは、`wandb_workspaces` APIを試すことができるように、新しい[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}})を作成します。

注：一意の`Saved view` URLを使用して、既存の[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})をロードできます。これを行う方法については、次のコードブロックを参照してください。

```python
# Weights & Biasesを初期化してログインします
wandb.login()

# 新しいプロジェクトを作成し、サンプルデータを記録する関数
def create_project_and_log_data():
    project = "workspace-api-example"  # デフォルトのプロジェクト名

    # サンプルデータを記録するためにrunを初期化します
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

# 新しいプロジェクトを作成し、データを記録します
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### （オプション）既存の[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}})と[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})をロードする
新しい[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}})を作成する代わりに、独自の既存の[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}})と[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})をロードできます。これを行うには、一意の[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}}) URLを見つけて、文字列として`ws.Workspace.from_url`に渡します。URLの形式は`https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc`です。

例：

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc").

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. プログラムによる[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})の例
以下は、プログラムによる[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})機能を使用する例です。

```python
# ワークスペース、セクション、およびパネルで利用可能なすべての設定を表示します。
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view`で[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})を作成する
この例では、新しい[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})を作成し、セクションとパネルを入力する方法を示します。[Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})は、通常のPythonオブジェクトのように編集でき、柔軟性と使いやすさを提供します。

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

### URLから[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})をロードする
元の設定に影響を与えることなく、[Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})を複製してカスタマイズします。これを行うには、既存の[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})をロードし、新しいビューとして保存します。

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save_as_new_view()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})の名前が「Updated Workspace Name」になっていることに注意してください。

### 基本設定
次のコードは、[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})を作成し、パネル付きのセクションを追加し、[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})、個々のセクション、およびパネルの設定を構成する方法を示しています。

```python
# カスタム設定でワークスペースを作成および構成する関数
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

# 関数を実行してワークスペースを作成および構成します
custom_settings_example(entity, project)
```

「An example workspace」という別の保存されたビューを表示していることに注意してください。

## [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をカスタマイズする
次のコードセルは、[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をプログラムでフィルタリング、色の変更、グループ化、および並べ替える方法を示しています。

各例では、一般的なワークフローは、`ws.RunsetSettings`の適切なパラメータへの引数として、目的のカスタマイズを指定することです。

### [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をフィルタリングする
Python式と、`wandb.log`で記録するメトリクス、または**Created Timestamp**のように[Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})の一部として自動的に記録されるメトリクスを使用してフィルタを作成できます。**Name**、**Tags**、または**ID**など、W&B App UIでの表示方法でフィルタを参照することもできます。

次の例は、検証損失の要約、検証精度の要約、および指定された正規表現に基づいて[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をフィルタリングする方法を示しています。

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # プロジェクト内のすべてのRunsを取得します
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 複数のフィルタを適用します：val_loss < 0.1、val_accuracy > 0.8、およびrun名が正規表現パターンと一致します
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss'サマリーでRunsをフィルタリングします
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy'サマリーでRunsをフィルタリングします
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # 's'で始まるrun名に一致するように正規表現検索を追加します
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

フィルタ式のリストを渡すと、ブール値の「AND」ロジックが適用されることに注意してください。

### [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})の色を変更する
この例では、[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})で[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})の色を変更する方法を示します。

```python
def run_color_example(entity: str, project: str) -> None:
    # プロジェクト内のすべてのRunsを取得します
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Runsに色を動的に割り当てます
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

### [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をグループ化する

この例では、特定のメトリクスで[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をグループ化する方法を示します。

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

### [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をソートする
この例では、検証損失の要約に基づいて[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})をソートする方法を示します。

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
            order=[ws.Ordering(ws.Summary("val_loss"))] # val_lossサマリーを使用して順序付けます
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. すべてをまとめる：包括的な例

この例では、包括的な[Workspace]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}})を作成し、その設定を構成し、セクションにパネルを追加する方法を示します。

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # プロジェクト内のすべてのRunsを取得します
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Runsに色を動的に割り当て、run設定を作成します
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