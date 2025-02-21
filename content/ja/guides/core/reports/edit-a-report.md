---
title: Edit a report
description: App UI を使用してインタラクティブに、または W&B SDK を使用してプログラムで、 レポート を編集します。
menu:
  default:
    identifier: ja-guides-core-reports-edit-a-report
    parent: reports
weight: 20
---

App UIまたは W&B SDKを使用して、インタラクティブに、またはプログラムでレポートを編集できます。

Reports は _ブロック_ で構成されています。ブロックはレポートの本文を構成します。これらのブロック内に、テキスト、画像、埋め込み 可視化 、実験と run からのプロット、および パネル グリッドを追加できます。

_パネル グリッド_ は、パネルと _run sets_ を保持する特定のタイプのブロックです。Run sets は、W&B の プロジェクト に ログ された run のコレクションです。パネル は run set データの 可視化 です。

{{% alert %}}
保存された ワークスペース ビューを作成およびカスタマイズする方法のステップバイステップの例については、[プログラムによる ワークスペース のチュートリアル]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) を確認してください。
{{% /alert %}}

{{% alert %}}
プログラムで レポート を編集する場合は、W&B Python SDKに加えて、`wandb-workspaces` がインストールされていることを確認してください。

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## プロット を追加する

各 パネル グリッドには、run sets のセットと パネル のセットがあります。セクションの下部にある run sets は、グリッド内の パネル に表示されるデータを制御します。別のセットの run からデータを取得するグラフを追加する場合は、新しい パネル グリッドを作成します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート にスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。**パネル を追加** を選択して、パネル を追加します。折れ線グラフ、散布図、平行座標グラフなど、W&B でサポートされている パネル を追加できます。

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="レポート にグラフを追加する" >}}
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}
SDKを使用して、プログラムで レポート に プロット を追加します。1つまたは複数の プロット またはグラフ オブジェクトのリストを `PanelGrid` Public API Class の `panels` パラメータに渡します。関連付けられた Python Class を使用して、 プロット またはグラフ オブジェクトを作成します。

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

プログラムで レポート に追加できる利用可能な プロット とグラフの詳細については、`wr.panels` を参照してください。

{{% /tab %}}
{{< /tabpane >}}

## Run sets を追加する

App UIまたは W&B SDKを使用して、 プロジェクト からインタラクティブに run sets を追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート にスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンから、パネル グリッドを選択します。これにより、レポート の作成元の プロジェクト から run set が自動的にインポートされます。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}

`wr.Runset()` および `wr.PanelGrid` Class を使用して、 プロジェクト から run sets を追加します。次の手順では、runset を追加する方法について説明します。

1. `wr.Runset()` オブジェクト インスタンスを作成します。project パラメータの runsets を含む プロジェクト の名前と、entity パラメータの プロジェクト を所有する エンティティ の名前を指定します。
2. `wr.PanelGrid()` オブジェクト インスタンスを作成します。1つまたは複数の runset オブジェクトのリストを `runsets` パラメータに渡します。
3. 1つまたは複数の `wr.PanelGrid()` オブジェクト インスタンスをリストに保存します。
4. レポート インスタンスの blocks 属性を パネル グリッド インスタンスのリストで更新します。

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

オプションで、SDKへの1回の呼び出しで runsets と パネル を追加できます。

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

レポート は run sets を自動的に更新して、 プロジェクト からの最新のデータを表示します。run set を *固定* することにより、レポート で run set を保持できます。run set を固定すると、ある時点での レポート 内の run set の状態が保持されます。

レポート の表示中に run set を固定するには、**Filter** ボタンの近くにある パネル グリッド内の雪片アイコンをクリックします。

{{< img src="/images/reports/freeze_runset.png" alt="" >}}

## コード ブロックを追加する

App UIまたは W&B SDKを使用して、 レポート に コード ブロックをインタラクティブに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート にスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンから **コード** を選択します。

コード ブロックの右側にある プログラミング言語の名前を選択します。これにより、ドロップダウンが展開されます。ドロップダウンから、 プログラミング言語の構文を選択します。Javascript、Python、CSS、JSON、HTML、Markdown、YAMLから選択できます。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wr.CodeBlock` Class を使用して、プログラムで コード ブロックを作成します。language パラメータと code パラメータに、それぞれ言語の名前と表示するコードを指定します。

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

これにより、次のような コード ブロックがレンダリングされます。

```yaml
this:
- is
- a
cool:
- yaml
- file
```

次の例は、Python コード ブロックを示しています。

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

これにより、次のような コード ブロックがレンダリングされます。

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## Markdownを追加する

App UIまたは W&B SDKを使用して、 レポート に Markdown をインタラクティブに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート にスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンから **Markdown** を選択します。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wandb.apis.reports.MarkdownBlock` Class を使用して、プログラムで Markdown ブロックを作成します。文字列を `text` パラメータに渡します。

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

## HTML要素を追加する

App UIまたは W&B SDKを使用して、 レポート に HTML 要素をインタラクティブに追加します。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート にスラッシュ (`/`) を入力して、ドロップダウン メニューを表示します。ドロップダウンから、テキスト ブロックのタイプを選択します。たとえば、H2 見出しブロックを作成するには、`Heading 2` オプションを選択します。

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

1つまたは複数の HTML 要素のリストを `wandb.apis.reports.blocks` 属性に渡します。次の例は、H1、H2、および順序なしリストを作成する方法を示しています。

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

これにより、次の HTML 要素がレンダリングされます。

{{< img src="/images/reports/render_html.png" alt="" >}}

{{% /tab %}}

{{% /tabpane %}}

## リッチ メディア リンクを埋め込む

App UIまたは W&B SDKを使用して、 レポート 内にリッチ メディアを埋め込みます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

URLをコピーして レポート に貼り付けて、 レポート 内にリッチ メディアを埋め込みます。次のアニメーションは、Twitter、YouTube、SoundCloudからURLをコピーして貼り付ける方法を示しています。

### Twitter

ツイート リンク URL をコピーして レポート に貼り付けると、 レポート 内でツイートを表示できます。

{{< img src="/images/reports/twitter.gif" alt="" >}}

### Youtube

YouTubeビデオ URL リンクをコピーして貼り付けて、ビデオを レポート に埋め込みます。

{{< img src="/images/reports/youtube.gif" alt="" >}}

### SoundCloud

SoundCloudリンクをコピーして貼り付けて、オーディオ ファイルを レポート に埋め込みます。

{{< img src="/images/reports/soundcloud.gif" alt="" >}}

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

1つまたは複数の埋め込みメディア オブジェクトのリストを `wandb.apis.reports.blocks` 属性に渡します。次の例は、ビデオとTwitterメディアを レポート に埋め込む方法を示しています。

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

## パネル グリッド を複製および削除する

再利用したいレイアウトがある場合は、 パネル グリッドを選択してコピーアンドペーストし、同じ レポート 内で複製したり、別の レポート に貼り付けたりすることもできます。

右上隅にあるドラッグ ハンドルを選択して、 パネル グリッド セクション全体を強調表示します。クリックしてドラッグし、 パネル グリッド、テキスト、見出しなど、 レポート 内の領域を強調表示して選択します。

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="" >}}

パネル グリッドを選択し、キーボードの `delete` を押して パネル グリッドを削除します。

{{< img src="/images/reports/delete_panel_grid.gif" alt="" >}}

## 見出しを折りたたんで Reports を整理する

レポート の見出しを折りたたんで、テキスト ブロック内のコンテンツを非表示にします。レポート がロードされると、展開されている見出しのみがコンテンツを表示します。レポート で見出しを折りたたむと、コンテンツを整理し、過剰なデータ ロードを防ぐことができます。次のgifは、その プロセス を示しています。

{{< img src="/images/reports/collapse_headers.gif" alt="" >}}
