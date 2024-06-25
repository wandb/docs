---
description: レポートを対話的にアプリUIで編集するか、W&B SDKを使用してプログラムで編集します。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Edit a report

<head>
  <title>Edit a W&B Report</title>
</head>

レポートを編集するには、App UI を使用する方法と、W&B SDK を使用する方法があります。

レポートは _ブロック_ で構成されています。ブロックはレポートの本文を作成します。これらのブロック内には、テキスト、画像、埋め込み可視化、Experiments や run のプロット、パネルグリッドを追加できます。

_パネルグリッド_ は、パネルと _run セット_ を保持する特定の種類のブロックです。Run セットは、W&B のプロジェクトにログされた run のコレクションです。パネルは、run セットデータの可視化です。

:::info
Python SDK を使用してプログラム的に Reports を編集する機能はベータ版であり、現在開発中です。
:::

### プロットを追加

各パネルグリッドには run セットとパネルのセットがあります。セクションの下にある run セットは、グリッド内のパネルに表示されるデータを制御します。異なる run セットからデータを取得するチャートを追加したい場合は、新しいパネルグリッドを作成します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポート内でスラッシュ (`/`) を入力すると、ドロップダウンメニューが表示されます。**Add panel** を選択してパネルを追加します。W&B がサポートする任意のパネル（線グラフ、散布図、並列座標チャートなど）を追加できます。

![Add charts to a report](/images/reports/demo_report_add_panel_grid.gif)
  
  </TabItem>
  <TabItem value="sdk">

SDK を使用してプログラム的にレポートにプロットを追加します。`PanelGrid` パブリック API クラスの `panels` パラメータにプロットまたはチャートオブジェクトのリストを渡します。それぞれの Python クラスを使ってプロットやチャートオブジェクトを作成します。

以下の例では、線グラフと散布図の作成方法を示しています。

```python
import wandb
import wandb.apis.reports as wr

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

レポートにプログラム的に追加できるプロットやチャートの詳細については、`wr.panels` を参照してください。
  </TabItem>
</Tabs>

### Run セットを追加

プロジェクトから run セットを App UI または W&B SDK を使用して追加します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポート内でスラッシュ (`/`) を入力してドロップダウンメニューを表示します。ドロップダウンから Panel Grid を選択します。これにより、レポートが作成されたプロジェクトから run セットが自動的にインポートされます。
  </TabItem>
  <TabItem value="sdk">

`wr.Runset()` および `wr.PanelGrid` クラスを使用してプロジェクトから run セットを追加します。以下の手順は run セットの追加方法を説明しています：

1. `wr.Runset()` オブジェクトインスタンスを作成します。プロジェクトパラメータには run セットを含むプロジェクトの名前を、エンティティパラメータにはプロジェクトを所有するエンティティの名前を指定します。
2. `wr.PanelGrid()` オブジェクトインスタンスを作成します。`runsets` パラメータに run セットオブジェクトのリストを渡します。
3. `wr.PanelGrid()` オブジェクトインスタンスをリストに格納します。
4. レポートインスタンスのブロック属性をパネルグリッドインスタンスのリストで更新します。

```python
import wandb
import wandb.apis.reports as wr

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

オプションで、SDK への 1 つの呼び出しで run セットとパネルを追加することができます：

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
  </TabItem>
</Tabs>

### コードブロックの追加

コードブロックをレポートに追加するには、App UI または W&B SDK を使用します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポート内でスラッシュ (`/`) を入力してドロップダウンメニューを表示します。ドロップダウンから **Code** を選択してください。

コードブロックの右側でプログラミング言語の名前を選択します。これによりドロップダウンが展開されます。ドロップダウンからプログラミング言語のシンタックスを選択します。Javascript、Python、CSS、JSON、HTML、Markdown、YAML から選べます。
  </TabItem>
  <TabItem value="sdk">

`wr.CodeBlock` クラスを使用してプログラム的にコードブロックを作成します。言語と表示したいコードの名前をそれぞれ language と code パラメータに指定します。

以下の例では、YAML ファイルのリストを示しています：

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    )
]

report.save()
```

これにより以下のようなコードブロックが表示されます：

```yaml
this:
- is
- a
cool:
- yaml
- file
```

次の例では、Python コードブロックを示しています：

```python
report = wr.Report(project="report-editing")

report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

これにより以下のようなコードブロックが表示されます：

```md
Hello, World!
```
  </TabItem>
</Tabs>

### Markdown

Markdown をレポートに追加するには、App UI または W&B SDK を使用します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポート内でスラッシュ (`/`) を入力してドロップダウンメニューを表示します。ドロップダウンから **Markdown** を選択してください。
  </TabItem>
  <TabItem value="sdk">

`wandb.apis.reports.MarkdownBlock` クラスを使用してプログラム的に Markdown ブロックを作成します。`text` パラメータに文字列を渡します：

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

これにより以下のような Markdown ブロックが表示されます：

![](/images/reports/markdown.png)
  </TabItem>
</Tabs>

### HTML 要素

HTML 要素をレポートに追加するには、App UI または W&B SDK を使用します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポート内でスラッシュ (`/`) を入力してドロップダウンメニューを表示します。ドロップダウンからテキストブロックの種類を選択します。例えば、H2 ヘディングブロックを作成するには、`Heading 2` オプションを選択します。
  </TabItem>
  <TabItem value="sdk">

1 つ以上の HTML 要素のリストを `wandb.apis.reports.blocks` 属性に渡します。以下の例では、H1、H2、および箇条書きリストの作成方法を示しています：

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.H1(text="How Programmatic Reports work"),
    wr.H2(text="Heading 2"),
    wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
]

report.save()
```

これにより以下のように HTML 要素が表示されます：

![](/images/reports/render_html.png)

  </TabItem>
</Tabs>

### リッチメディアリンクの埋め込み

レポートにリッチメディアを埋め込むには、App UI または W&B SDK を使用します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポートに URL をコピー＆ペーストしてリッチメディアを埋め込みます。以下のアニメーションは、Twitter、YouTube、SoundCloud からの URL をコピー＆ペーストする手順を示しています。

#### Twitter

レポートにツイートのリンク URL をコピー＆ペーストすると、レポート内にツイートが表示されます。

![](/images/reports/twitter.gif)

#### Youtube

YouTube 動画の URL リンクをコピー＆ペーストしてレポートに動画を埋め込みます。

![](/images/reports/youtube.gif)

#### SoundCloud

SoundCloud のリンクをコピー＆ペーストしてレポートに音声ファイルを埋め込みます。

![](/images/reports/soundcloud.gif)
  </TabItem>
  <TabItem value="sdk">

1 つ以上の埋め込みメディアオブジェクトのリストを `wandb.apis.reports.blocks` 属性に渡します。以下の例は、動画と Twitter メディアをレポートに埋め込む方法を示しています：

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc%5Etfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>\n'
    ),
]
report.save()
```
  </TabItem>
</Tabs>

### パネルグリッドの重複と削除

レイアウトを再利用したい場合は、パネルグリッド全体を選択してコピー＆ペーストし、同じレポート内で重複させたり、別のレポートにペーストしたりできます。

右上のドラッグハンドルを選択して、パネルグリッドセクション全体をハイライトします。クリックしてドラッグすると、レポート内のパネルグリッド、テキスト、ヘディングなどの領域がハイライトされます。

![](/images/reports/demo_copy_and_paste_a_panel_grid_section.gif)

パネルグリッドを選択し、キーボードの `delete` を押してパネルグリッドを削除します。

![](@site/static/images/reports/delete_panel_grid.gif)

### レポートを整理するためのヘッダーの折りたたみ

ヘッダーを折りたたむことで、レポート内のテキストブロックの内容を非表示にできます。レポートが読み込まれると、展開されたヘッダーのみが内容を表示します。レポート内のヘッダーを折りたたむことで、内容を整理し、過剰なデータの読み込みを防ぐことができます。以下の gif はそのプロセスを示しています。

![](@site/static/images/reports/collapse_headers.gif)