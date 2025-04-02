---
title: Edit a report
description: App UI を使用してインタラクティブに、または W&B SDK を使用してプログラム的に レポート を編集します。
menu:
  default:
    identifier: ja-guides-core-reports-edit-a-report
    parent: reports
weight: 20
---

App UIまたは W&B SDKでプログラム的に、インタラクティブにレポートを編集できます。

Reports は _ブロック_ で構成されています。ブロックはレポートの本文を構成します。これらのブロック内には、テキスト、画像、埋め込み 可視化、実験からのプロットと run、および パネル グリッドを追加できます。

_パネル グリッド_ は、パネルと _run sets_ を保持する特定のタイプのブロックです。Run sets は、W&B のプロジェクトに記録された runs の集合です。パネルは run set データの 可視化です。

{{% alert %}}
保存されたワークスペース ビューを作成およびカスタマイズする方法のステップバイステップの例については、[プログラムによるワークスペースのチュートリアル]({{< relref path="/tutorials/workspaces.md" lang="ja" >}})をご覧ください。
{{% /alert %}}

{{% alert %}}
レポートをプログラムで編集する場合は、W&B Python SDKに加えて `wandb-workspaces` がインストールされていることを確認してください。

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## プロットを追加する

各パネル グリッドには、run sets のセットとパネルのセットがあります。セクションの下部にある run sets は、グリッド内のパネルに表示されるデータを制御します。異なる runs のセットからデータを取得するグラフを追加する場合は、新しいパネル グリッドを作成します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートにスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。[**パネルを追加**] を選択して、パネルを追加します。折れ線グラフ、散布図、平行座標グラフなど、W&B でサポートされている任意のパネルを追加できます。

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="Add charts to a report" >}}
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}
SDKを使用して、プログラムでプロットをレポートに追加します。1つ以上のプロットまたはグラフ オブジェクトのリストを `PanelGrid` Public API Classの `panels` パラメータに渡します。関連するPythonクラスを使用してプロットまたはグラフ オブジェクトを作成します。

次の例は、折れ線グラフと散布図を作成する方法を示しています。

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

プログラムでレポートに追加できる利用可能なプロットとグラフの詳細については、`wr.panels` を参照してください。

{{% /tab %}}
{{< /tabpane >}}

## Run sets を追加する

App UIまたは W&B SDKを使用して、プロジェクトからrun setsをインタラクティブに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートにスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンから、[パネル グリッド] を選択します。これにより、レポートが作成されたプロジェクトから run set が自動的にインポートされます。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}

`wr.Runset()` および `wr.PanelGrid` クラスを使用して、プロジェクトから run sets を追加します。次の手順では、run set を追加する方法について説明します。

1. `wr.Runset()` オブジェクト インスタンスを作成します。プロジェクトの run sets が含まれるプロジェクトの名前を project パラメータに、プロジェクトを所有するエンティティを entity パラメータに指定します。
2. `wr.PanelGrid()` オブジェクト インスタンスを作成します。1つ以上の run set オブジェクトのリストを `runsets` パラメータに渡します。
3. 1つ以上の `wr.PanelGrid()` オブジェクト インスタンスをリストに格納します。
4. パネル グリッド インスタンスのリストで report インスタンス ブロック属性を更新します。

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

オプションで、SDKへの1回の呼び出しで run sets とパネルを追加できます。

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

## Run set を固定する

レポートは、プロジェクトからの最新のデータを表示するために、run sets を自動的に更新します。Run set を*固定* することで、レポート内の run set を保持できます。Run set を固定すると、ある時点でのレポート内の run set の状態が保持されます。

レポートの表示中に run set を固定するには、[フィルタ] ボタンの近くにあるパネル グリッド内の雪のアイコンをクリックします。

{{< img src="/images/reports/freeze_runset.png" alt="" >}}

## コード ブロックを追加する

App UIまたは W&B SDKを使用して、インタラクティブにコード ブロックをレポートに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートにスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンから [**コード**] を選択します。

コード ブロックの右側にあるプログラミング言語の名前を選択します。これにより、ドロップダウンが展開されます。ドロップダウンから、プログラミング言語の構文を選択します。Javascript、Python、CSS、JSON、HTML、Markdown、および YAML から選択できます。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wr.CodeBlock` クラスを使用して、プログラムでコード ブロックを作成します。それぞれ言語パラメータとコード パラメータに表示する言語の名前とコードを指定します。

たとえば、次の例は YAML ファイルのリストを示しています。

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

これにより、次のようなコード ブロックがレンダリングされます。

```yaml
this:
- is
- a
cool:
- yaml
- file
```

次の例は、Pythonコード ブロックを示しています。

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

これにより、次のようなコード ブロックがレンダリングされます。

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## Markdown を追加する

App UIまたは W&B SDKを使用して、インタラクティブに Markdown をレポートに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートにスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンから [**Markdown**] を選択します。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wandb.apis.reports.MarkdownBlock` クラスを使用して、プログラムで Markdown ブロックを作成します。文字列を `text` パラメータに渡します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

これにより、次のような Markdown ブロックがレンダリングされます。

{{< img src="/images/reports/markdown.png" alt="" >}}

{{% /tab %}}

{{% /tabpane %}}

## HTML 要素を追加する

App UIまたは W&B SDKを使用して、インタラクティブに HTML 要素をレポートに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートにスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンからテキスト ブロックのタイプを選択します。たとえば、H2 見出しブロックを作成するには、[見出し 2] オプションを選択します。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

1つ以上のHTML要素のリストを `wandb.apis.reports.blocks` 属性に渡します。次の例は、H1、H2、および順序なしリストを作成する方法を示しています。

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

これにより、HTML要素が次のようにレンダリングされます。

{{< img src="/images/reports/render_html.png" alt="" >}}

{{% /tab %}}

{{% /tabpane %}}

## リッチ メディア リンクを埋め込む

App UIまたは W&B SDKを使用して、レポート内にリッチ メディアを埋め込みます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

URLをコピーしてレポートに貼り付け、レポート内にリッチ メディアを埋め込みます。次のアニメーションは、Twitter、YouTube、および SoundCloud から URL をコピーして貼り付ける方法を示しています。

### Twitter

ツイート リンクの URL をコピーしてレポートに貼り付け、レポート内でツイートを表示します。

{{< img src="/images/reports/twitter.gif" alt="" >}}

### Youtube

YouTube ビデオの URL リンクをコピーして貼り付け、ビデオをレポートに埋め込みます。

{{< img src="/images/reports/youtube.gif" alt="" >}}

### SoundCloud

SoundCloud リンクをコピーして貼り付け、オーディオ ファイルをレポートに埋め込みます。

{{< img src="/images/reports/soundcloud.gif" alt="" >}}

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

1つ以上の埋め込みメディア オブジェクトのリストを `wandb.apis.reports.blocks` 属性に渡します。次の例は、ビデオと Twitter メディアをレポートに埋め込む方法を示しています。

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

## パネル グリッドを複製および削除する

再利用したいレイアウトがある場合は、パネル グリッドを選択してコピーして貼り付け、同じレポートに複製したり、別のレポートに貼り付けたりすることもできます。

右上隅にあるドラッグ ハンドルを選択して、パネル グリッド セクション全体を強調表示します。クリックしてドラッグして、パネル グリッド、テキスト、見出しなどのレポート内の領域を強調表示して選択します。

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="" >}}

パネル グリッドを選択し、キーボードの `delete` を押してパネル グリッドを削除します。

{{< img src="/images/reports/delete_panel_grid.gif" alt="" >}}

## ヘッダーを折りたたんで Reports を整理する

Report のヘッダーを折りたたんで、テキスト ブロック内のコンテンツを非表示にします。レポートが読み込まれると、展開されているヘッダーのみがコンテンツを表示します。レポートでヘッダーを折りたたむと、コンテンツを整理し、過剰なデータ読み込みを防ぐことができます。次の gif は、そのプロセスを示しています。

{{< img src="/images/reports/collapse_headers.gif" alt="Collapsing headers in a report." >}}

## 複数の次元にわたる関係を可視化する

複数の次元にわたる関係を効果的に 可視化 するには、カラー グラデーションを使用して変数の1つを表します。これにより、明瞭さが向上し、パターンが解釈しやすくなります。

1. カラー グラデーションで表す変数を選択します (例: ペナルティ スコア、学習率など)。これにより、トレーニング時間 (x軸) にわたって、ペナルティ (色) が報酬/副作用 (y軸) とどのように相互作用するかをより明確に理解できます。
2. 主要な傾向を強調表示します。特定の runs のグループにカーソルを合わせると、 可視化 でそれらが強調表示されます。
