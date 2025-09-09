---
title: Report を編集する
description: App UI で対話的に、または W&B SDK を使ってプログラムからレポートを編集できます。
menu:
  default:
    identifier: ja-guides-core-reports-edit-a-report
    parent: reports
weight: 20
---

{{% alert %}}
W&B Report and Workspace API はパブリックプレビューです。
{{% /alert %}}

report は W&B App の UI で対話的に、または W&B SDK でプログラムから編集できます。

Reports は _blocks_ から構成されます。blocks は report の本文を形作ります。各 block には、テキストや画像、埋め込みの可視化、Experiments や run のプロット、パネルグリッドを追加できます。

_パネルグリッド_ はパネルと _run set_ を保持する特別な種類の block です。run set は W&B の project にログされた runs の集合です。パネルは run set のデータを可視化したものです。


{{% alert %}}
保存済みの Workspace ビューを作成・カスタマイズする手順は、[Programmatic Workspaces チュートリアル]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{% alert %}}
レポートをプログラムから編集する場合は、W&B Python SDK に加えて W&B Report and Workspace API `wandb-workspaces` がインストールされていることを確認してください:

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## プロットを追加する

各パネルグリッドは run set の集合とパネルの集合を持ちます。セクション下部の run set が、そのグリッド内のパネルに表示されるデータを制御します。別の runs の集合からデータを取得するチャートを追加したい場合は、新しいパネルグリッドを作成してください。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

report 内でスラッシュ（`/`）を入力するとドロップダウンメニューが表示されます。そこから **Add panel** を選んでパネルを追加します。折れ線、散布図、パラレルコーディネートなど、W&B がサポートしている任意のパネルを追加できます。

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="レポートにチャートを追加" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
SDK を使ってプログラムからレポートにプロットを追加します。`PanelGrid` Public API クラスの `panels` パラメータに、1 つ以上のプロットまたはチャートオブジェクトのリストを渡します。各プロットやチャートのオブジェクトは対応する Python クラスで作成します。

以下の例では、折れ線プロットと散布図の作成方法を示します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

blocks = [
    wr.PanelGrid(
        panels=[
            wr.LinePlot(x="time", y="velocity"),
            wr.ScatterPlot(x="time", y="acceleration"),
        ]
    )
]

report.blocks = blocks
report.save()
```

レポートにプログラムから追加できるプロットやチャートの詳細は `wr.panels` を参照してください。

{{% /tab %}}
{{< /tabpane >}}


## run set を追加する

run set は W&B App の UI から対話的に、または W&B SDK から追加できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

report 内でスラッシュ（`/`）を入力するとドロップダウンメニューが表示されます。ドロップダウンから **Panel Grid** を選びます。これにより、その report が作成された project から自動的に run set が取り込まれます。

パネルを report にインポートすると、run 名は project から継承されます。必要に応じて report 内で読者に文脈を与えるために [run の名前を変更]({{< relref path="/guides/models/track/runs/#rename-a-run" lang="ja" >}}) できます。run の名前変更は個々のパネルの中だけで行われます。同じ report 内でそのパネルをクローンした場合は、クローンしたパネルでも run 名が変更されます。

1. report 内で鉛筆アイコンをクリックしてレポートエディタを開きます。
1. run set で、名前を変更したい run を探します。report 名の上にカーソルを置き、縦三点リーダーをクリックします。次のいずれかを選び、フォームを送信します。

    - **Rename run for project**: project 全体でその run の名前を変更します。新しいランダム名を生成するには、フィールドを空のままにします。
    - **Rename run for panel grid**: その report の中だけで run 名を変更し、他のコンテキストでは既存の名前を保持します。新しいランダム名の生成はサポートしていません。

1. **Publish report** をクリックします。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

`wr.Runset()` と `wr.PanelGrid` クラスを使って project から run set を追加します。以下は run set を追加する手順です:

1. `wr.Runset()` のオブジェクトを作成します。project パラメータには run set を含む project 名を、entity パラメータにはその project を所有する entity 名を指定します。
2. `wr.PanelGrid()` のオブジェクトを作成します。`run sets` パラメータに 1 つ以上の run set オブジェクトのリストを渡します。
3. 1 つ以上の `wr.PanelGrid()` オブジェクトをリストに格納します。
4. report インスタンスの blocks 属性に、そのパネルグリッドのリストを設定します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")]
)

report.blocks = [panel_grids]
report.save()
```

1 回の呼び出しで run set とパネルを同時に追加することもできます:

```python
import wandb

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    panels=[
        wr.LinePlot(
            title="line title",
            x="x",
            y=["y"],
            range_x=[0, 100],
            range_y=[0, 100],
            log_x=True,
            log_y=True,
            title_x="x axis title",
            title_y="y axis title",
            ignore_outliers=True,
            groupby="hyperparam1",
            groupby_aggfunc="mean",
            groupby_rangefunc="minmax",
            smoothing_factor=0.5,
            smoothing_type="gaussian",
            smoothing_show_original=True,
            max_runs_to_show=10,
            plot_type="stacked-area",
            font_size="large",
            legend_position="west",
        ),
        wr.ScatterPlot(
            title="scatter title",
            x="y",
            y="y",
            # z='x',
            range_x=[0, 0.0005],
            range_y=[0, 0.0005],
            # range_z=[0,1],
            log_x=False,
            log_y=False,
            # log_z=True,
            running_ymin=True,
            running_ymean=True,
            running_ymax=True,
            font_size="small",
            regression=True,
        ),
    ],
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")],
)


report.blocks = [panel_grids]
report.save()
``` 

{{% /tab %}}
{{< /tabpane >}}


## run set を固定する

report は project の最新データを表示するように run set を自動更新します。report 内の run set をある時点の状態のまま残したい場合は、その run set を「固定（freeze）」できます。run set を固定すると、その時点での run set の状態を report 内に保存します。

report を表示中に run set を固定するには、パネルグリッドの **Filter** ボタン近くにある雪の結晶アイコンをクリックします。

{{< img src="/images/reports/freeze_runset.png" alt="Runset を固定するボタン" >}}

## プログラムから run set をグループ化する

[Workspace and Reports API]({{< relref path="/ref/python/wandb_workspaces/reports" lang="ja" >}}) を使って、プログラムから run set 内の runs をグループ化できます。

run set 内の runs は、config 値、run メタデータ、またはサマリーメトリクスでグループ化できます。以下の表に、利用可能なグルーピング方法と、その際に利用可能なキーを示します:

| グルーピング方法 | 説明 | 利用可能なキー |
| ---|------| --- |
| Config values| config 値で runs をグループ化 | `wandb.init(config=)` の config パラメータで指定した値 |
| Run metadata| run のメタデータで runs をグループ化 | `State`, `Name`, `JobType` |
| Summary metrics| サマリーメトリクスで runs をグループ化 | `wandb.Run.log()` で run にログした値 |




### config 値でグループ化する

似た設定の runs を比較するために、config 値でグループ化します。config 値は run の設定（`wandb.init(config=)`）で指定するパラメータです。config 値でグループ化するには、`config.<key>` の構文を使います。ここで `<key>` はグループ化対象の config 名です。 

例えば、次のコードスニペットでは、まず `group` の config 値で run を初期化し、その後 report 内で `group` の config 値に基づいて runs をグループ化します。`<entity>` と `<project>` はあなたの W&B の entity と project 名に置き換えてください。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

entity = "<entity>"
project = "<project>"

for group in ["control", "experiment_a", "experiment_b"]:
    for i in range(3):
        with wandb.init(entity=entity, project=project, group=group, config={"group": group, "run": i}, name=f"{group}_run_{i}") as run:
            # 簡単なトレーニングのシミュレーション
            for step in range(100):
                run.log({
                    "acc": 0.5 + (step / 100) * 0.3 + (i * 0.05),
                    "loss": 1.0 - (step / 100) * 0.5
                })
```

Python のスクリプトやノートブック内で、`config.group` の値で runs をグループ化できます:

```python
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["config.group"]  # "group" の config 値でグループ化
)
```

続けて、グループ化した run set を用いて report を作成できます:

```python
report = wr.Report(
  entity=entity,
  project=project,
  title="Grouped Runs Example",
)

report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
          )
      ]

report.save()
```

### run メタデータでグループ化する

run の名前（`Name`）、状態（`State`）、ジョブタイプ（`JobType`）で runs をグループ化できます。 

上の例に続けて、次のコードスニペットで run 名でグループ化できます:

```python
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["Name"]  # run 名でグループ化
)
```

{{% alert %}}
run の名前は `wandb.init(name=)` パラメータで指定した名前です。名前を指定しなかった場合、W&B がランダムな名前を生成します。

run の名前は W&B App の run の **Overview** ページ、またはプログラムから `Api.runs().run.name` で確認できます。
{{% /alert %}}

### サマリーメトリクスでグループ化する

以下の例は、サマリーメトリクスで runs をグループ化する方法を示します。サマリーメトリクスは `wandb.Run.log()` で run にログした値です。run をログした後は、W&B App の run の **Overview** ページ内 **Summary** セクションでサマリーメトリクス名を確認できます。

サマリーメトリクスでグループ化する構文は `summary.<key>` です。ここで `<key>` はグループ化対象のサマリーメトリクス名です。 

例えば、`acc` というサマリーメトリクスをログしたとします:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

entity = "<entity>"
project = "<project>"

for group in ["control", "experiment_a", "experiment_b"]:
    for i in range(3):
        with wandb.init(entity=entity, project=project, group=group, config={"group": group, "run": i}, name=f"{group}_run_{i}") as run:
            # 簡単なトレーニングのシミュレーション
            for step in range(100):
                run.log({
                    "acc": 0.5 + (step / 100) * 0.3 + (i * 0.05),
                    "loss": 1.0 - (step / 100) * 0.5
                })

```

その後、`summary.acc` で runs をグループ化できます:

```python
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["summary.acc"]  # サマリーの値でグループ化 
)
```

## プログラムから run set をフィルタする

[Workspace and Reports API]({{< relref path="/ref/python/wandb_workspaces/reports" lang="ja" >}}) を使って、プログラムから run set をフィルタし、report に追加できます。

一般的なフィルタ式の構文は次のとおりです:

```text
Filter('key') operation <value>
```

ここで `key` はフィルタ名、`operation` は比較演算子（例: `>`, `<`, `==`, `in`, `not in`, `or`, `and`）、`<value>` は比較対象の値です。`Filter` は適用したいフィルタの種類のプレースホルダーです。利用可能なフィルタと説明は以下のとおりです:

| フィルタ | 説明 | 利用可能なキー |
| ---|---| --- |
|`Config('key')` | config 値でフィルタ | `wandb.init(config=)` の `config` パラメータで指定した値 |
|`SummaryMetric('key')` | サマリーメトリクスでフィルタ | `wandb.Run.log()` で run にログした値 |
|`Tags('key')` | タグでフィルタ | run に（プログラムから、または W&B App で）追加したタグの値 |
|`Metric('key')` | run のプロパティでフィルタ | `tags`, `state`, `displayName`, `jobType` |

フィルタを定義したら、`wr.PanelGrid(runsets=)` にフィルタ済みの run set を渡して report を作成できます。本ページの各所にある **Report and Workspace API** タブで、プログラムからレポートにさまざまな要素を追加する方法を参照してください。

以下の例は、report 内で run set をフィルタする方法を示します。`<>` で囲まれた値は適宜置き換えてください。

### Config フィルタ

run set を 1 つ以上の config 値でフィルタします。config 値は run の設定（`wandb.init(config=)`）で指定するパラメータです。

例えば、次のコードスニペットでは、まず `learning_rate` と `batch_size` の config 値で run を初期化し、その後 `learning_rate` の config 値に基づいて report 内の runs をフィルタします。

```python
import wandb

config = {
    "learning_rate": 0.01,
    "batch_size": 32,
}

with wandb.init(project="<project>", entity="<entity>", config=config) as run:
    # ここにトレーニングコードを記述
    pass
```

Python のスクリプトやノートブック内で、学習率が `0.01` より大きい runs をプログラムからフィルタできます。

```python
import wandb_workspaces.reports.v2 as wr

runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Config('learning_rate') > 0.01"
)
```

`and` 演算子を使って複数の config 値でフィルタすることもできます:
 
```python
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Config('learning_rate') > 0.01 and Config('batch_size') == 32"
)
```

前の例に続けて、次のようにフィルタ済みの run set を使って report を作成できます:

```python
report = wr.Report(
  entity="<entity>",
  project="<project>",
  title="My Report"
)

report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
      panels=[
          wr.LinePlot(
              x="Step",
              y=["accuracy"],
          )
      ]
  )
]

report.save()
```

### Metric フィルタ

run のタグ（`tags`）、run の状態（`state`）、run 名（`displayName`）、ジョブタイプ（`jobType`）で run set をフィルタします。

{{% alert %}}
`Metric` フィルタは異なる構文を取ります。値の集合はリストとして渡します。

```text
Metric('key') operation [<value>]
```
{{% /alert %}}

例えば、次の Python スニペットは 3 つの run を作成し、それぞれに名前を割り当てます:

```python
import wandb

with wandb.init(project="<project>", entity="<entity>") as run:
    for i in range(3):
        run.name = f"run{i+1}"
        # ここにトレーニングコードを記述
        pass
```

report を作成する際に、表示名で runs をフィルタできます。例えば、`run1`、`run2`、`run3` という名前の runs をフィルタするには、次のコードを使います:

```python
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('displayName') in ['run1', 'run2', 'run3']"
)
```

{{% alert %}}
run の名前は W&B App の run の **Overview** ページ、またはプログラムから `Api.runs().run.name` で確認できます。
{{% /alert %}}

以下は run の状態（`finished`、`crashed`、`running`）で run set をフィルタする例です:

```python
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('state') in ['finished']"
)
```

```python
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('state') not in ['crashed']"
)
```


### SummaryMetric フィルタ

以下はサマリーメトリクスで run set をフィルタする例です。サマリーメトリクスは `wandb.Run.log()` で run にログした値です。run をログした後は、W&B App の run の **Overview** ページ内 **Summary** セクションでサマリーメトリクス名を確認できます。

```python
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="SummaryMetric('accuracy') > 0.9"
)
```

```python
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('state') in ['finished'] and SummaryMetric('train/train_loss') < 0.5"
)
```

### Tags フィルタ

次のコードスニペットは、タグで run set をフィルタする方法を示します。タグは（プログラムから、または W&B App で）run に追加する値です。

```python
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Tags('training') == 'training'"
)
```

## コードブロックを追加する

コードブロックは W&B App の UI から、または W&B SDK を使って report に追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

report 内でスラッシュ（`/`）を入力するとドロップダウンメニューが表示されます。ドロップダウンから **Code** を選びます。

コードブロック右側のプログラミング言語名を選択します。ドロップダウンが開くので、シンタックスハイライトする言語を選びます。JavaScript、Python、CSS、JSON、HTML、Markdown、YAML から選択できます。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wr.CodeBlock` クラスを使って、プログラムからコードブロックを作成します。language と code パラメータに、それぞれ言語名と言語に対応したコードを指定します。

例えば、次の例は YAML ファイルのリストを示します:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    )
]

report.save()
```

これは次のようなコードブロックとして表示されます:

```yaml
this:
- is
- a
cool:
- yaml
- file
```

次の例は Python のコードブロックです:

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

これは次のようなコードブロックとして表示されます:

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## Markdown を追加する

Markdown は W&B App の UI から、または W&B SDK を使って report に追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

report 内でスラッシュ（`/`）を入力するとドロップダウンメニューが表示されます。ドロップダウンから **Markdown** を選びます。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.MarkdownBlock` クラスを使って、プログラムから markdown ブロックを作成します。`text` パラメータに文字列を渡します:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

これは次のような markdown ブロックとして表示されます:

{{< img src="/images/reports/markdown.png" alt="Markdown ブロックのレンダリング" >}}

{{% /tab %}}

{{% /tabpane %}}


## HTML 要素を追加する

HTML 要素は W&B App の UI から、または W&B SDK を使って report に追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

report 内でスラッシュ（`/`）を入力するとドロップダウンメニューが表示されます。ドロップダウンからテキストブロックの種類を選択します。例えば、H2 の見出しブロックを作成するには、`Heading 2` を選びます。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.blocks` 属性に 1 つ以上の HTML 要素のリストを渡します。以下は H1、H2、箇条書きを作成する例です:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.H1(text="How Programmatic Reports work"),
    wr.H2(text="Heading 2"),
    wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
]

report.save()
```

これは次のように HTML 要素として表示されます:


{{< img src="/images/reports/render_html.png" alt="HTML 要素のレンダリング" >}}

{{% /tab %}}

{{% /tabpane %}}

## リッチメディアリンクを埋め込む

W&B App の UI から、または W&B SDK を使って、report にリッチメディアを埋め込めます。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

URL を report にコピー＆ペーストすると、リッチメディアを埋め込めます。以下のアニメーションは、Twitter、YouTube、SoundCloud から URL をコピー＆ペーストする手順を示します。

### Twitter

Tweet のリンク URL を report に貼り付けると、report 内でその Tweet を表示できます。

{{< img src="/images/reports/twitter.gif" alt="Twitter コンテンツの埋め込み" >}}

### Youtube

YouTube の動画 URL を貼り付けると、report に動画を埋め込めます。

{{< img src="/images/reports/youtube.gif" alt="YouTube 動画の埋め込み" >}}

### SoundCloud

SoundCloud のリンクを貼り付けると、report に音声ファイルを埋め込めます。

{{< img src="/images/reports/soundcloud.gif" alt="SoundCloud オーディオの埋め込み" >}}

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.blocks` 属性に 1 つ以上の埋め込みメディアオブジェクトのリストを渡します。以下は、動画と Twitter のメディアを report に埋め込む例です:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc%5Etfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>\n'
    ),
]
report.save()
```

{{% /tab %}}

{{% /tabpane %}}

## パネルグリッドの複製と削除

再利用したいレイアウトがある場合は、パネルグリッドを選択してコピー＆ペーストすることで、同じ report 内に複製したり、別の report に貼り付けることができます。

右上のドラッグハンドルを選択して、パネルグリッド全体のセクションをハイライトします。クリック＆ドラッグで、パネルグリッド、テキスト、見出しなど、report 内の領域をハイライトして選択できます。

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="パネルグリッドのコピー" >}}

パネルグリッドを選択した状態で、キーボードの `delete` を押すと削除できます。

{{< img src="/images/reports/delete_panel_grid.gif" alt="パネルグリッドの削除" >}}

## Reports を整理するために見出しを折りたたむ

Report 内の見出しを折りたたむと、テキストブロック内のコンテンツを非表示にできます。report の読み込み時には、展開されている見出しだけがコンテンツを表示します。report の見出しを折りたたむことで、コンテンツを整理し、不要なデータの読み込みを防げます。以下の GIF に手順を示します。

{{< img src="/images/reports/collapse_headers.gif" alt="report 内で見出しを折りたたむ" >}}

## 多次元にわたる関係を可視化する

多次元にわたる関係を効果的に可視化するには、変数の 1 つを色のグラデーションで表現します。視認性が高まり、パターンを解釈しやすくなります。

1. 色のグラデーションで表現する変数（例: 罰則スコア、学習率 など）を選びます。これにより、トレーニング時間（x 軸）に対して、報酬/副作用（y 軸）と罰則（色）がどのように相互作用するかをより明確に理解できます。
2. 主要なトレンドを強調します。特定の run のグループにカーソルを合わせると、可視化内でそれらがハイライト表示されます。