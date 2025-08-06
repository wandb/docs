---
title: プログラムによるワークスペース
menu:
  tutorials:
    identifier: workspaces
    parent: null
weight: 5
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb" >}}
機械学習実験をより効果的に整理・可視化するために、ワークスペースをプログラムで作成・管理・カスタマイズしましょう。`wandb-workspaces` W&B ライブラリを使えば、設定を定義し、パネルレイアウトを設定し、セクションを整理することができます。ワークスペースを URL で読み込み・編集したり、式を使って run をフィルタやグループ化したり、run の見た目をカスタマイズしたりできます。

`wandb-workspaces` は、W&B の [Workspaces]({{< relref "/guides/models/track/workspaces/" >}}) と [Reports]({{< relref "/guides/core/reports/" >}}) をプログラムで作成・カスタマイズするための Python ライブラリです。

このチュートリアルでは、`wandb-workspaces` を使って設定の定義、パネルレイアウトの設定、セクションの整理を通してワークスペースを作成・カスタマイズする方法を紹介します。

## このノートブックの使い方
* 各セルを順番に実行してください。
* セル実行後に表示される URL をコピー＆ペーストすると、ワークスペースへの変更内容を確認できます。

{{% alert %}}
ワークスペースのプログラムによる操作は、現在 [Saved workspaces views]({{< relref "/guides/models/track/workspaces#saved-workspace-views" >}}) でサポートされています。Saved workspace view は、ワークスペースのコラボレーション用スナップショットです。チームメンバーであれば誰でも、Saved workspace view を閲覧・編集・保存できます。
{{% /alert %}}

## 1. 依存関係のインストールとインポート

```python
# 依存パッケージのインストール
!pip install wandb wandb-workspaces rich
```

```python
# 必要なモジュールのインポート
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # パネル追加のために Reports API を使用

# 出力フォーマットの改善
%load_ext rich
```

## 2. 新しい Project と Workspace を作成する

このチュートリアルでは、新しい Project を作成し、`wandb_workspaces` API を使って実験を行います。

※ 既存のワークスペースを `Saved view` の URL で読み込む方法については、次のコードブロックを参照してください。

```python
# W&B の初期化とログイン
wandb.login()

# 新しい Project を作成 & サンプルデータをログする関数
def create_project_and_log_data():
    project = "workspace-api-example"  # デフォルトの Project 名

    # サンプルデータをログするため、run を初期化
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

# 新しい Project を作成しデータを記録
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### （オプション）既存の Project と Workspace を読み込む
新しく Project を作成する代わりに、自分の既存 Project および Workspace を読み込むこともできます。その場合は、ユニークなワークスペースの URL を調べて、`ws.Workspace.from_url` の引数に文字列で渡します。URL の形は `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc` です。

例:

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc")

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. プログラムによるワークスペース操作例

下記は、プログラムでワークスペース操作ができる主なユースケース例です。

```python
# ワークスペース・セクション・パネルそれぞれの利用可能な設定一覧を確認
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### `saved view` を持つワークスペースを作成
新しいワークスペースを作成し、セクションやパネルを配置する例です。ワークスペースは通常の Python オブジェクトのように編集できるので、柔軟かつ簡単に操作できます。

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
既存のワークスペースを複製しカスタマイズできます（オリジナルには影響しません）。既存ワークスペースを読み込み、新しいビューとして保存します。

```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save_as_new_view()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

これで Workspace の名前が "Updated Workspace Name" に更新されました。

### 基本設定
以下のコードでは、ワークスペースを作成し、パネル付きセクションの追加や、ワークスペース・セクション・パネルごとの設定を行う方法を紹介しています。

```python
# カスタム設定のワークスペースを作成・設定する関数
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

# 関数を実行しカスタムワークスペースを作成・設定
custom_settings_example(entity, project)
```

この操作で "An example workspace" というSaved viewが作成されています。

## Run のカスタマイズ
以下のコードセルでは、Run のフィルタ、色の変更、グループ化、ソートをプログラムで指定する方法を紹介します。

いずれの例でも、`ws.RunsetSettings` の対応パラメータに希望のカスタマイズを引数で指定します。

### Run のフィルタ
`wandb.log` でログしたメトリクスや自動で記録される **Created Timestamp** などで Python 式を使いフィルタできます。W&B App の UI で表示される **Name**、**Tags**、**ID** も指定できます。

例では、検証ロス、検証精度、および指定した正規表現でフィルタしています。

```python
def advanced_filter_example(entity: str, project: str) -> None:
    # プロジェクト内の全 Run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # 複数のフィルタを適用: val_loss < 0.1, val_accuracy > 0.8, run名が正規表現に一致
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
                (ws.Summary("val_loss") < 0.1),  # 'val_loss' サマリーでフィルタ
                (ws.Summary("val_accuracy") > 0.8),  # 'val_accuracy' サマリーでフィルタ
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # run名が's'から始まるものを正規表現で検索
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

フィルタ式をリストで渡すと、論理積（AND）で実行されます。

### Run の色変更
以下は、ワークスペースの各 run の色を変更する例です。

```python
def run_color_example(entity: str, project: str) -> None:
    # プロジェクト内の全 Run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # run に色を割り当てる
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

以下は、特定のメトリクスで run をグループ化する例です。

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

### Run のソート
検証ロス（val_loss）でRunをソートする例です。

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
            order=[ws.Ordering(ws.Summary("val_loss"))] # val_lossサマリーでソート
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. すべてを組み合わせた総合例

この例では、ワークスペースの包括的な作成から、設定・セクション・パネル追加まで一連の操作を示します。

```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # プロジェクト内の全 Run を取得
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Run に動的に色を割り当て、run 設定を作成
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