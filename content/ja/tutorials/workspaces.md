---
title: プログラマティック ワークスペース
menu:
  tutorials:
    identifier: ja-tutorials-workspaces
    parent: null
weight: 5
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
機械学習実験をより効果的に整理し可視化するために、プログラムからワークスペースを作成・管理・カスタマイズできます。設定を定義したり、パネルのレイアウトを設定したり、セクションを整理することが [`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B ライブラリで可能です。ワークスペースの URL で読み込み・編集ができ、式を使って run をフィルタ・グループ化したり、run の見た目をカスタマイズできます。

`wandb-workspaces` は、W&B の [Workspaces]({{< relref path="/guides/models/track/workspaces/" lang="ja" >}}) や [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) をプログラムから作成・カスタマイズするための Python ライブラリです。

このチュートリアルでは、`wandb-workspaces` を使って設定を定義し、パネルレイアウトやセクション整理を行い、ワークスペースを作成・カスタマイズする方法を紹介します。

## このノートブックの使い方
* セルを一つずつ実行してください。
* 各セルを実行後に表示される URL をコピー＆ペーストして、ワークスペースの変更内容を確認することができます。


{{% alert %}}
ワークスペースとのプログラム的なやり取りは、現在 [保存済みワークスペースビュー]({{< relref path="/guides/models/track/workspaces#saved-workspace-views" lang="ja" >}}) に対応しています。保存済みワークスペースビューは、ワークスペースの共同編集できるスナップショットです。チームの誰でもその保存済みワークスペースビューを閲覧・編集・保存可能です。
{{% /alert %}}

## 1. 依存ライブラリのインストールとインポート


```python
# 依存ライブラリのインストール
!pip install wandb wandb-workspaces rich
```


```python
# 依存ライブラリのインポート
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # パネル追加のため Reports API を利用

# 出力フォーマットを改善
%load_ext rich
```

## 2. 新しい Project・Workspace の作成

このチュートリアルでは、新しい Project を作成し、`wandb_workspaces` API を使って実験します。

※ 既存のワークスペースは、その一意な `Saved view` URL を使って読み込むこともできます。詳しくは次のコードブロックをご覧ください。


```python
# W&B を初期化してログイン
wandb.login()

# 新しいプロジェクトを作成し、サンプルデータをログする関数
def create_project_and_log_data():
    project = "workspace-api-example"  # デフォルトのプロジェクト名

    # サンプルデータをログする run を初期化
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

# 新しいプロジェクトを作成し、データをログ
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### （オプション）既存の Project と Workspace を読み込む
新しい Project を作成せずに、自分の既存 Project・Workspace を読み込むこともできます。その場合、ワークスペースの一意な URL を `ws.Workspace.from_url` の文字列として渡してください。URL は `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc` の形式です。

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

## 3. プログラムによるワークスペース構築例
以下はプログラムからワークスペースの機能を使うサンプルです：


```python
# ワークスペース、セクション、パネルで使える全設定項目を確認
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view` でワークスペースを作成
この例では、新規のワークスペースを作成し、セクションやパネルで構成します。Python のオブジェクトのようにワークスペースを柔軟に編集できます。


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
もとの設定に影響を与えず、ワークスペースを複製・カスタマイズ可能です。その場合、既存のワークスペースを読み込んで新しいビューとして保存します：


```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save_as_new_view()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

この手順でワークスペース名が "Updated Workspace Name" となります。

### 基本設定
次のコードはワークスペースを作成し、パネル付きセクションを追加し、ワークスペースや各セクション・パネルの設定をカスタマイズする方法を示します：


```python
# カスタム設定付きワークスペースを作成・設定する関数
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

# この関数でワークスペースを作成・設定します
custom_settings_example(entity, project)
```

この手順で、"An example workspace" という名前の異なる保存ビューを表示することになります。

## run のカスタマイズ
次のコードでは、run をフィルタしたり色やグループ分け、ソート順などをプログラムで設定する方法を紹介します。

いずれも基本的なワークフローは、`ws.RunsetSettings` の各パラメータに目的のカスタマイズを引数で指定することです。

### run のフィルタ
Python 式や `wandb.log` で記録したメトリクス、または各 run で自動記録された **Created Timestamp** などを用いて柔軟にフィルタを作成できます。また W&B の UI 上に表示される **Name**、**Tags**、**ID** でフィルタも可能です。

例では、検証損失・検証精度・正規表現を使ってフィルタします：


```python
def advanced_filter_example(entity: str, project: str) -> None:
    # プロジェクト内の全 run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 複数フィルタ適用: val_loss < 0.1, val_accuracy > 0.8, run 名が正規表現一致
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' summary で run をフィルタ
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' summary で run をフィルタ
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # run 名が s で始まるものを正規表現で検索
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

フィルタ式リストを渡すと、論理積（AND）条件となります。

### run の色を変更
以下の例で、ワークスペース内の各 run に色を指定する方法を示します：


```python
def run_color_example(entity: str, project: str) -> None:
    # プロジェクト内の全 run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run に色を動的割り当て
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

### run をグループ化

この例では、指定したメトリクスで run をグループ化する方法を示します。



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

### run をソート
この例では検証損失 summary によって run をソートする方法を示します：


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
            order=[ws.Ordering(ws.Summary("val_loss"))] # val_loss summary でソート
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. 総合例：すべてをまとめて使う

この例は設定をカスタマイズし、複数パネルを持つセクションも作成した包括的なワークスペースの使い方を示します：


```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # プロジェクト内の全 run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run に色を割り当て、run 設定を作成
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