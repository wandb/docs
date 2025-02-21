---
title: Programmatic Workspaces
menu:
  tutorials:
    identifier: ja-tutorials-workspaces
    parent: null
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
機械学習実験をより効果的に整理し、可視化するために、プログラムで作成、管理、カスタマイズできるワークスペースを使いましょう。設定を定義したり、パネルレイアウトを設定したり、セクションを整理したりできます。[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B ライブラリを利用して、URLを用いたワークスペースの読み込みや変更、実験のフィルタリングやグループ化、見た目のカスタマイズが可能です。

`wandb-workspaces` は W&B [Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}}) や [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) をプログラムで作成およびカスタマイズするための Python ライブラリです。

このチュートリアルでは、`wandb-workspaces` を利用してワークスペースを作成およびカスタマイズする方法を学びます。設定の定義、パネルレイアウトの設定、セクションの整理が含まれます。

## このノートブックの使い方
* セルを1つずつ実行してください。
* セルを実行した後に表示される URL をコピーし、ワークスペースに加えた変更を確認してください。

{{% alert %}}
ワークスペースとのプログラムによる相互作用は、現在 [**保存されたワークスペースビュー**]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ja" >}}) でサポートされています。保存されたワークスペースビューは、ワークスペースの共同スナップショットです。チームの誰でも保存されたワークスペースビューを表示、編集、保存できます。 
{{% /alert %}}

## 1. 必要なものをインストールしてインポートする

```python
# 必要なものをインストール
!pip install wandb wandb-workspaces rich
```

```python
# 必要なものをインポート
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # パネルを追加するために Reports API を使用

# 出力形式を改善
%load_ext rich
```

## 2. 新しいプロジェクトとワークスペースを作成する

このチュートリアルでは、新しいプロジェクトを作成し、`wandb_workspaces` API で実験します。

注: 既存のワークスペースは、そのユニークな `Saved view` URL を使用して読み込むことができます。次のコードブロックでその方法を確認します。

```python
# Weights & Biases を初期化してログイン
wandb.login()

# 新しいプロジェクトを作成し、サンプルデータをログする関数を定義
def create_project_and_log_data():
    project = "workspace-api-example"  # デフォルトのプロジェクト名

    # サンプルデータをログするために run を初期化
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

# 新しいプロジェクトを作成し、データをログ
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### (オプション) 既存のプロジェクトとワークスペースを読み込む
新しいプロジェクトを作成する代わりに、既存のプロジェクトとワークスペースを読み込むことができます。これを行うには、ユニークなワークスペース URL を見つけて `ws.Workspace.from_url` に文字列として渡します。URL は `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc` という形式です。

例:

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc").

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. プログラムでのワークスペースの例
以下は、プログラムでのワークスペース機能を使用する例です:

```python
# ワークスペース、セクション、パネルのすべての設定を確認
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view` を使ったワークスペースの作成
この例は、新しいワークスペースを作成してセクションとパネルで埋める方法を示しています。ワークスペースは通常の Python オブジェクトのように編集できますので、柔軟で使いやすいです。

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

### URL からワークスペースを読み込む
元の設定に影響を与えることなく、ワークスペースを複製およびカスタマイズします。これを行うには、既存のワークスペースを読み込み、新しいビューとして保存します:

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

ワークスペースが「Updated Workspace Name」として保存されたことを確認してください。

### 基本設定
次のコードは、ワークスペースを作成し、パネルを追加したセクションを設定し、ワークスペース、個々のセクション、およびパネルの設定を構成する方法を示します:

```python
# カスタム設定を持つワークスペースを作成および設定する関数
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

# ワークスペースを作成および設定する関数を実行
custom_settings_example(entity, project)
```

「An example workspace」という別の保存されたビューを閲覧していることに注意してください。

## run のカスタマイズ
次のコードセルでは、run をプログラムでフィルタリング、色の変更、グループ化、および並べ替えする方法を示します。

各例では、一般的なワークフローとして、適切な `ws.RunsetSettings` のパラメータに引数としてカスタマイズを指定します。

### run のフィルタリング
`wandb.log` でログしたことや、run の一部として自動的にログされたメトリクスを使用して、Python 式でフィルタを作成できます。また、W&B App UI での表示方法（**Name**、**Tags**、**ID** など）でフィルタを参照できます。

次の例では、検証損失のサマリー、検証精度のサマリー、および指定された正規表現に基づいて run をフィルタリングする方法を示しています:

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # プロジェクト内のすべての run の取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 複数のフィルタを適用: val_loss < 0.1, val_accuracy > 0.8, および run 名が正規表現に一致
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' サマリーで run をフィルタ
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' サマリーで run をフィルタ
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # 正規表現検索を追加して、's' で始まる run 名を一致させる
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

フィルタ式のリストを渡すことで、ブール "AND" ロジックが適用されることに注意してください。

### run の色を変更する
この例は、ワークスペース内の run の色を変更する方法を示しています:

```python
def run_color_example(entity: str, project: str) -> None:
    # プロジェクト内のすべての run の取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run に色を動的に割り当てる
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

### run のグループ化

この例は、特定のメトリクスで run をグループ化する方法を示しています。

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

### run の並べ替え
この例は、検証損失のサマリーに基づいて run を並べ替える方法を示しています:

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
            order=[ws.Ordering(ws.Summary("val_loss"))] #val_loss サマリーによる順番で並べ替え
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. 全体をまとめる: 包括的な例

この例は、包括的なワークスペースを作成し、その設定を構成し、セクションにパネルを追加する方法を示しています:

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # プロジェクト内のすべての run の取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run に色を動的に割り当て、run 設定を作成
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