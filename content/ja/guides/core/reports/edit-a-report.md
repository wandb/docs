---
title: Edit a report
description: レポートをインタラクティブにアプリ UI で編集するか、W&B SDK を使用してプログラムから編集します。
menu:
  default:
    identifier: ja-guides-core-reports-edit-a-report
    parent: reports
weight: 20
---

レポートを App UI で対話的に、または W&B SDK を使ってプログラム的に編集します。

Reports は _ブロック_ で構成されています。ブロックはレポートの本文を構成します。これらのブロックには、テキスト、画像、埋め込み可視化、実験と run のプロット、パネル グリッドを追加できます。

_パネル グリッド_ はパネルと _run セット_ を持つ特定のタイプのブロックです。Run セットとは、W&B のプロジェクトにログされた run のコレクションのことです。パネルは run セット データの可視化です。

{{% alert %}}
レポートを保存されたワークスペースビューとして作成し、カスタマイズする方法についてのステップバイステップの例は、[ワークスペースのプログラム的チュートリアル]({{< relref path="/tutorials/workspaces.md" lang="ja" >}})をご覧ください。
{{% /alert %}}

{{% alert %}}
レポートをプログラム的に編集したい場合は、W&B Python SDK に加えて `wandb-workspaces` がインストールされていることを確認してください。

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## プロットを追加する

各パネル グリッドには、run セットとパネルのセットがあります。セクションの下部にある run セットは、グリッド内のパネルに表示されるデータを制御します。異なる run セットからデータを取得してグラフを追加したい場合は、新しいパネル グリッドを作成します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートでスラッシュ (`/`)を入力すると、ドロップダウンメニューが表示されます。 **Add panel** を選択してパネルを追加します。W&B がサポートするラインプロット、散布プロット、またはパラレル座標チャートなど、任意のパネルを追加できます。

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="レポートにチャートを追加する" >}}
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}
SDK を使用してレポートにプロットをプログラム的に追加します。`PanelGrid` Public API クラスの `panels` パラメータに 1 つ以上のプロットまたはチャートオブジェクトのリストを渡します。関連する Python クラスを使ってプロットまたはチャート オブジェクトを作成します。

次の例では、ラインプロットと散布プロットの作成方法を示します。

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

プログラム的にレポートに追加できる利用可能なプロットとチャートの詳細については、 `wr.panels` を参照してください。

{{% /tab %}}
{{< /tabpane >}}


## Run セットを追加する

Run セットを W&B SDK または App UI を使ってプロジェクトから対話的に追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートでスラッシュ (`/`)を入力するとドロップダウンメニューが表示されます。ドロップダウンから Panel Grid を選択します。これにより、レポートが作成されたプロジェクトから自動的に run セットがインポートされます。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}

`wr.Runset()` および `wr.PanelGrid` クラスを使用してプロジェクトから run セットを追加します。次の手順では、runset の追加方法について説明します。

1. `wr.Runset()` オブジェクトインスタンスを作成します。プロジェクトパラメータにrunsetを含むプロジェクトの名前を、entityパラメータにプロジェクトを所有する entity を指定します。
2. `wr.PanelGrid()` オブジェクトインスタンスを作成します。1 つ以上の runset オブジェクトを `runsets` パラメータに渡します。
3. 1 つ以上の `wr.PanelGrid()` オブジェクトインスタンスをリストに保存します。
4. パネル グリッド インスタンスのリストでレポート インスタンス ブロック属性を更新します。

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

SDK への 1 回の呼び出しで runset と パネルを追加することもできます。

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

## Run セットをフリーズする

プロジェクトから最新のデータを表示するために、レポートは自動的に run セットを更新します。*Freezing* された run セットは、レポート内の run セットの状態をある時点で保持します。

レポートを表示するときに run セットをフリーズするには、**Filter** ボタンの近くにあるパネル グリッド内のスノーフレーク アイコンをクリックします。

{{< img src="/images/reports/freeze_runset.png" alt="" >}}

## コードブロックを追加する

コードブロックを App UI または W&B SDK を使用して対話的にレポートに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートでスラッシュ (`/`)を入力すると、ドロップダウンメニューが表示されます。 ドロップダウンから**Code**を選択します。

コードブロックの右側にあるプログラミング言語の名前を選択します。これによりドロップダウンが展開されます。ドロップダウンからプログラミング言語の構文を選択します。選択可能なのは Javascript, Python, CSS, JSON, HTML, Markdown, YAML です。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wr.CodeBlock` クラスを使って、コードブロックをプログラム的に作成します。language パラメータには言語の名前を、code パラメータには表示したいコードを指定します。

以下の例は YAML ファイルのリストを示しています。

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

これにより、次のようなコードブロックがレンダリングされます。

```yaml
this:
- is
- a
cool:
- yaml
- file
```

次の例は Python コードブロックです。

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

これにより、次のようなコードブロックがレンダリングされます。

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## Markdown を追加する

Markdown を App UI または W&B SDK を使用して対話的にレポートに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート内でスラッシュ (`/`) を入力すると、ドロップダウンメニューが表示されます。 ドロップダウンから**Markdown**を選択します。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wandb.apis.reports.MarkdownBlock` クラスを使って、markdown ブロックをプログラム的に作成します。`text` パラメータには文字列を渡します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

これにより、次のような markdown ブロックがレンダリングされます。

{{< img src="/images/reports/markdown.png" alt="" >}}

{{% /tab %}}

{{% /tabpane %}}


## HTML 要素を追加する

HTML 要素を App UI または W&B SDK を使用して対話的にレポートに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートでスラッシュ (`/`)を入力すると、ドロップダウンメニューが表示されます。ドロップダウンからテキストブロックのタイプを選択します。たとえば、H2 見出しブロックを作成するには、`Heading 2` オプションを選択します。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wandb.apis.reports.blocks` 属性に 1 つ以上の HTML 要素を渡します。次の例では、H1、H2、および順序なしリストを作成する方法を示します。

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

これにより、次のような HTML 要素が表示されます。

{{< img src="/images/reports/render_html.png" alt="" >}}

{{% /tab %}}

{{% /tabpane %}}

## リッチメディアリンクを埋め込む

リッチメディアを App UI または W&B SDK を使ってレポート内に埋め込みます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

URL をコピーしてレポート内に貼り付けることで、リッチメディアをレポート内に埋め込むことができます。次のアニメーションでは、Twitter、YouTube、SoundCloud から URL をコピーして貼り付ける方法を示しています。

### Twitter

Tweet のリンク URL をレポートにコピーアンドペーストして、レポート内で Tweet を表示します。

{{< img src="/images/reports/twitter.gif" alt="" >}}

### YouTube

YouTube ビデオの URL リンクをコピーアンドペーストして、レポート内にビデオを埋め込みます。

{{< img src="/images/reports/youtube.gif" alt="" >}}

### SoundCloud

SoundCloud リンクをコピーアンドペーストして、レポートに音声ファイルを埋め込みます。

{{< img src="/images/reports/soundcloud.gif" alt="" >}}

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

1つ以上の埋め込みメディアオブジェクトのリストを `wandb.apis.reports.blocks` 属性に渡します。次の例では、ビデオと Twitter メディアをレポートに埋め込む方法を示します。

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

再利用したいレイアウトがある場合は、パネル グリッドを選択し、コピーして貼り付けて、同じレポート内または別のレポートに複製できます。

右上隅のドラッグハンドルを選択して、パネルグリッドセクション全体をハイライトします。クリックしてドラッグすると、レポート内のパネルグリッド、テキスト、見出しなどの領域をハイライトして選択できます。

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="" >}}

パネルグリッドを選択し、キーボードの `delete` キーを押してパネルグリッドを削除します。

{{< img src="/images/reports/delete_panel_grid.gif" alt="" >}}

## レポートを整理するためにヘッダーを折りたたむ

レポート内のテキストブロックのコンテンツを非表示にするために、レポート内のヘッダーを折りたたみます。レポートが読み込まれると、展開されているヘッダーだけがコンテンツを表示します。レポート内のヘッダーを折りたたむと、コンテンツを整理し、データの過剰な読み込みを防ぐのに役立ちます。次の gif では、そのプロセスを示しています。

{{< img src="/images/reports/collapse_headers.gif" alt="" >}}