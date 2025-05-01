---
title: ワークスペースのプログラム管理
menu:
  tutorials:
    identifier: ja-tutorials-workspaces
    parent: null
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
プログラムでワークスペースを作成、管理、カスタマイズすることにより、機械学習実験をより効果的に整理し、可視化できます。[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B ライブラリを使用して設定を定義し、パネルのレイアウトを設定し、セクションを整理できます。URLでワークスペースを読み込み、変更したり、runをフィルタリングしてグループ化するための式を使用したり、runの外観をカスタマイズすることも可能です。

`wandb-workspaces` は、W&Bの [Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}}) と [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) をプログラムで作成しカスタマイズするためのPythonライブラリです。

このチュートリアルでは、`wandb-workspaces` を使って設定を定義し、パネルレイアウトを設定し、セクションを整理することでワークスペースを作成しカスタマイズする方法を学びます。

## このノートブックの使い方
* 各セルを一度に1つずつ実行してください。
* セルの実行後に表示されるURLをコピーして貼り付けることで、ワークスペースへの変更を確認できます。

{{% alert %}}
プログラムによるワークスペースとの対話は、現在[**Saved workspaces views**]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ja" >}})でサポートされています。Saved workspaces viewsは、ワークスペースの協働スナップショットです。同じチームのメンバーであれば、誰でも保存されたワークスペースビューを表示、編集、保存できます。
{{% /alert %}}

## 1. 依存関係のインストールとインポート

```python
# 依存関係のインストール
!pip install wandb wandb-workspaces rich
```

```python
# 依存関係のインポート
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # パネルを追加するためのReports APIを使用します

# 出力フォーマットを改善
%load_ext rich
```

## 2. 新しいプロジェクトとワークスペースの作成

このチュートリアルでは、新しいプロジェクトを作成し、`wandb_workspaces` APIを使用して実験を行います。

注：ユニークな `Saved view` URLを使用して既存のワークスペースを読み込むことができます。この方法は次のコードブロックで説明しています。

```python
# Weights & Biases を初期化してログイン
wandb.login()

# 新しいプロジェクトとサンプルデータをログに記録するための関数
def create_project_and_log_data():
    project = "workspace-api-example"  # デフォルトのプロジェクト名

    # サンプルデータをログに記録するためにrunを初期化
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

# 新しいプロジェクトを作成してデータをログ
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### （オプション）既存のプロジェクトとワークスペースを読み込む

新しいプロジェクトを作成する代わりに、既存のプロジェクトとワークスペースを読み込むことができます。そのためには、ユニークなワークスペースURLを見つけて、それを文字列として `ws.Workspace.from_url` に渡します。URLの形式は `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc` です。

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

## 3. プログラムによるワークスペースの例

以下に、プログラムによるワークスペースの特徴を使用するための例を示します：

```python
# ワークスペース、セクション、およびパネルに利用可能なすべての設定を確認します。
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view` を使ってワークスペースを作成

この例では、新しいワークスペースを作成し、それにセクションとパネルを配置する方法を示します。ワークスペースは通常のPythonオブジェクトのように編集できますので、柔軟性と使いやすさを提供します。

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

### URLからワークスペースを読み込む

元のセットアップに影響を与えずにワークスペースを複製してカスタマイズします。これを行うためには、既存のワークスペースを読み込み、新しいビューとして保存します。

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save_as_new_view()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

今、あなたのワークスペースの名前は "Updated Workspace Name" です。

### 基本設定

次のコードは、ワークスペースを作成し、パネルを追加して、ワークスペース、個々のセクション、およびパネルの設定を構成する方法を示します。

```python
# カスタム設定でワークスペースを作成して構成するための関数
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

# ワークスペースを作成して設定する関数を実行
custom_settings_example(entity, project)
```

今、あなたは "An example workspace" という名前の別の保存済みビューを見ています。

## Run のカスタマイズ

次のコードセルでは、runをフィルタリングし、色を変え、グループ化し、プログラムで並べ替える方法を示します。

各例では、一般的なワークフローとして、適切なパラメータに引数として指定されたカスタマイズ情報を `ws.RunsetSettings` に渡します。

### Run のフィルタリング

Pythonの式と `wandb.log` でログしたメトリクス、または **Created Timestamp** のようにrunの一部として自動的にログされるメトリクスを使って、フィルターを作成できます。また、W&B App UI での表示方法、たとえば **Name**、**Tags**、または **ID** に基づいてフィルターを参照することもできます。

次の例では、検証損失のサマリ、検証精度のサマリ、および指定された正規表現に基づいてrunをフィルタリングする方法を示します。

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # プロジェクト内のすべてのrunを取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 複数のフィルターを適用：val_loss < 0.1、val_accuracy > 0.8、およびrun名が正規表現に一致
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' のサマリでrunをフィルタ
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' のサマリでrunをフィルタ
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # run名が 's' で始まるものと一致するように正規表現検索を追加します
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

フィルター式のリストを渡すと、ブールの「AND」論理が適用されることに注意してください。

### Run の色を変更

この例では、ワークスペース内のrunの色を変更する方法を示します。

```python
def run_color_example(entity: str, project: str) -> None:
    # プロジェクト内のすべてのrunを取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Runの色を動的に割り当てる
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

### Run のグループ化

この例では、特定のメトリクスでrunをグループ化する方法を示します。

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

### Run をソート

この例では、検証損失のサマリに基づいてrunをソートする方法を示します。

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
            order=[ws.Ordering(ws.Summary("val_loss"))] #val_lossサマリを使用して注文
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. まとめ: 包括的な例

この例では、包括的なワークスペースを作成し、その設定を構成し、セクションにパネルを追加する方法を示します。

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # プロジェクト内のすべてのrunを取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Runの色を動的に割り当ててrunの設定を作成する
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