---
title: Programmatic Workspaces
menu:
  tutorials:
    identifier: ja-tutorials-workspaces
    parent: null
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
プログラムで _Workspaces_ を作成、管理、カスタマイズすることで、 機械学習 の 実験管理 をより効果的に整理し、 可視化 します。 設定 を定義し、 パネル のレイアウトを設定し、 [`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B ライブラリ でセクションを整理できます。 URL で _Workspaces_ をロードおよび変更したり、式を使用して _Runs_ をフィルタリングおよびグループ化したり、 _Runs_ の外観をカスタマイズしたりできます。

`wandb-workspaces` は、W&B [Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}}) および [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) をプログラムで作成およびカスタマイズするための Python ライブラリです。

この チュートリアル では、`wandb-workspaces` を使用して、 設定 を定義し、 パネル レイアウトを設定し、セクションを編成することで、 _Workspaces_ を作成およびカスタマイズする方法を説明します。

## この ノートブック の使用方法
* 各セルを一度に 1 つずつ実行します。
* セルを実行した後に表示される URL をコピーして貼り付け、 _Workspace_ に加えられた変更を表示します。

{{% alert %}}
_Workspaces_ とのプログラムによるインタラクションは、現在、[**保存された _Workspace_ ビュー**]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ja" >}}) でサポートされています。 保存された _Workspace_ ビューは、 _Workspace_ のコラボレーション スナップショット です。 チームの誰でも、保存された _Workspace_ ビューを表示、編集、変更を保存できます。
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
import wandb_workspaces.reports.v2 as wr # パネルを追加するために Reports API を使用します

# 出力形式を改善する
%load_ext rich
```

## 2. 新しい プロジェクト と _Workspace_ を作成する

この チュートリアル では、`wandb_workspaces` API を試すことができるように、新しい プロジェクト を作成します。

注: 一意の [Saved view] URL を使用して、既存の _Workspace_ をロードできます。 これを行う方法については、次の コード ブロック を参照してください。

```python
# Weights & Biases を初期化してログインする
wandb.login()

# 新しいプロジェクトを作成し、サンプルデータを記録する関数
def create_project_and_log_data():
    project = "workspace-api-example"  # デフォルトのプロジェクト名

    # サンプルデータを記録するために run を初期化する
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

# 新しいプロジェクトを作成し、データを記録する
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### (オプション) 既存の プロジェクト と _Workspace_ をロードする
新しい プロジェクト を作成する代わりに、独自の既存の プロジェクト と _Workspace_ をロードできます。 これを行うには、一意の _Workspace_ URL を見つけて、文字列として `ws.Workspace.from_url` に渡します。 URL の形式は `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc` です。

次に例を示します。

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc").

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. プログラムによる _Workspace_ の例
以下は、プログラムによる _Workspace_ 機能を使用する例です。

```python
# _Workspaces_ 、セクション、およびパネルで使用可能なすべての 設定 を表示します。
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### [Saved view] で _Workspace_ を作成する
この例では、新しい _Workspace_ を作成し、セクションと パネル を追加する方法を示します。 _Workspaces_ は通常の Python オブジェクト のように編集できるため、柔軟性と使いやすさが向上します。

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

### URL から _Workspace_ をロードする
元の 設定 に影響を与えずに、 _Workspaces_ を複製してカスタマイズします。 これを行うには、既存の _Workspace_ をロードし、新しいビューとして保存します。

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

_Workspace_ の名前が "Updated Workspace Name" になっていることに注意してください。

### 基本 設定
次の コード は、 _Workspace_ を作成し、 パネル を含むセクションを追加し、 _Workspace_ 、個々のセクション、および パネル の 設定 を 構成 する方法を示しています。

```python
# カスタム設定で _Workspace_ を作成および構成する関数
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

# 関数を実行して _Workspace_ を作成および構成する
custom_settings_example(entity, project)
```

別の保存されたビュー「An example workspace」を表示していることに注意してください。

## _Runs_ のカスタマイズ
次の コード セルは、プログラムで _Runs_ をフィルタリング、色の変更、グループ化、およびソートする方法を示しています。

各例では、一般的な ワークフロー は、`ws.RunsetSettings` の適切な パラメータ への 引数 として、目的のカスタマイズを指定することです。

### _Runs_ のフィルタリング
Python 式と、`wandb.log` で記録する メトリクス 、または **作成タイムスタンプ** など、 _Run_ の一部として自動的に記録される メトリクス で フィルタ を作成できます。 **名前** 、 **タグ** 、 **ID** など、W&B App UI に表示される方法で フィルタ を参照することもできます。

次の例は、 検証 損失 サマリー、 検証 精度 サマリー、および指定された正規表現に基づいて _Runs_ をフィルタリングする方法を示しています。

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # プロジェクト内のすべての _Runs_ を取得する
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 複数の フィルタ を適用する: val_loss < 0.1、val_accuracy > 0.8、および run 名が正規表現パターンに一致する
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' サマリーで _Runs_ をフィルタリングする
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' サマリーで _Runs_ をフィルタリングする
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # 's' で始まる run 名に一致するように正規表現検索を追加する
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

フィルタ 式のリストを渡すと、ブール値「AND」ロジックが適用されることに注意してください。

### _Runs_ の色を変更する
この例では、 _Workspace_ で _Runs_ の色を変更する方法を示します。

```python
def run_color_example(entity: str, project: str) -> None:
    # プロジェクト内のすべての _Runs_ を取得する
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # _Runs_ に色を動的に割り当てる
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

### _Runs_ のグループ化

この例では、特定の メトリクス で _Runs_ をグループ化する方法を示します。

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

### _Runs_ のソート
この例では、 検証 損失 サマリーに基づいて _Runs_ をソートする方法を示します。

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
            order=[ws.Ordering(ws.Summary("val_loss"))] #val_loss サマリーを使用して順序付けする
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. すべてをまとめる: 包括的な例

この例では、包括的な _Workspace_ を作成し、その 設定 を 構成 し、セクションに パネル を追加する方法を示します。

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # プロジェクト内のすべての _Runs_ を取得する
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # _Runs_ に色を動的に割り当て、run 設定 を作成する
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