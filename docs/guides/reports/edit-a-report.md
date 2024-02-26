---
description: >-
  Edit a report interactively with the App UI or programmatically with the
  Weights & Biases SDK.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# レポートの編集

<head>
  <title>W&Bレポートの編集</title>
</head>


App UI または Weights & Biases SDK を使って、レポートを対話的に編集します。

レポートは _ブロック_ で構成されています。ブロックはレポートの本文を作成します。これらのブロックには、テキスト、画像、埋め込まれた可視化、実験や run からのプロット、パネルグリッドを追加できます。

_パネルグリッド_ は、パネルと _runセット_ を保持する特定のタイプのブロックです。Runセットは、W&Bのプロジェクトにログされたrunのコレクションです。パネルは、runセットデータの可視化です。

:::caution
Python SDK を使ったプログラムによるレポート編集は、ベータ版で進行中の開発です。
:::

### プロットの追加

各パネルグリッドには、runセットとパネルのセットがあります。セクション下部のrunセットは、グリッド内のパネルに表示されるデータを制御します。異なるrunセットからデータを取得するチャートを追加したい場合は、新しいパネルグリッドを作成してください。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポート中にスラッシュ(`/`)を入力すると、ドロップダウンメニューが表示されます。**Add panel**を選択してパネルを追加します。Weights & Biasesでサポートされているパネルを追加できます。これには、折れ線グラフ、散布図、並行座標グラフなどが含まれます。

![レポートにグラフを追加](/images/reports/demo_report_add_panel_grid.gif)
  
  </TabItem>
  <TabItem value="sdk">

SDKを使用してレポートにプログラムでプロットを追加します。`PanelGrid` Public APIクラスの`panels`パラメーターに、1つ以上のプロットまたはチャートオブジェクトのリストを渡します。関連するPythonクラスでプロットまたはチャートオブジェクトを作成します。

次の例では、折れ線グラフと散布図を作成する方法を示しています。

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(
    project='report-editing',
    title='素晴らしいタイトル',
    description='説明的な説明。'
)

blocks = [
	wr.PanelGrid(
		panels=[
			wr.LinePlot(x="time", y="velocity"),
			wr.ScatterPlot(x="time", y="acceleration")
		]
	)
]
```

report.blocks = blocks
report.save()
```

利用可能なプロットやチャートをプログラムでレポートに追加する方法についての詳細は、`wr.panels`を参照してください。
  </TabItem>
</Tabs>


### ランセットを追加

App UIまたはWeights & Biases SDKを使って、プロジェクトからインタラクティブにランセットを追加します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポート内でスラッシュ（`/`）を入力すると、ドロップダウンメニューが表示されます。ドロップダウンからPanel Gridを選択すると、レポートが作成されたプロジェクトからランセットが自動的にインポートされます。
  </TabItem>
  <TabItem value="sdk">

`wr.Runset()` および `wr.PanelGrid` クラスを使用して、プロジェクトからランセットを追加します。以下の手順でランセットを追加する方法を説明します。

1. `wr.Runset()` オブジェクトのインスタンスを作成します。プロジェクトパラメータにランセットが含まれるプロジェクトの名前を指定し、エンティティパラメータにプロジェクトの所有者であるエンティティを指定します。
2. `wr.PanelGrid()` オブジェクトのインスタンスを作成します。`runsets` パラメータに1つ以上のランセットオブジェクトのリストを渡します。
3. 1つ以上の `wr.PanelGrid()` オブジェクトのインスタンスをリストに格納します。
4. リスト内のパネルグリッドインスタンスでレポートインスタンスのブロック属性を更新します。

```python
import wandb
import wandb.apis.reports as wr

レポート = wr.Report(
    project='report-editing',
    title='素晴らしいタイトル',
    description='説明的な説明。'
)

panel_grids = wr.PanelGrid(
    runsets=[wr.Runset(project='<プロジェクト名>', entity='<エンティティ名>')]
)

report.blocks = [panel_grids]
report.save()
```

必要に応じて、SDKへの1回の呼び出しでrunsetsとパネルを追加できます：

```python
import wandb
report = wr.Report(
    project='report-editing',
    title='素晴らしいタイトル',
    description='説明的な説明。'
)

panel_grids = wr.PanelGrid(
        panels=[
            wr.LinePlot(
                title="線のタイトル",
                x="x",
                y=["y"],
                range_x=[0, 100],
                range_y=[0, 100],
                log_x=True,
                log_y=True,
                title_x="x軸のタイトル",
                title_y="y軸のタイトル",
                ignore_outliers=True,
                groupby='hyperparam1',
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
                title="散布図のタイトル",
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
            )
				],
	runsets=[wr.Runset(project='<プロジェクト名>', entity='<エンティティ名>')]
		)

report.blocks = [panel_grids]
report.save()
```
  </TabItem>
</Tabs>


### コードブロックの追加

App UIまたはWeights & Biases SDKを使用して、レポートにコードブロックをインタラクティブに追加します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポートで `/` (フォワードスラッシュ) を入力すると、ドロップダウンメニューが表示されます。ドロップダウンから **Code** を選択します。

コードブロックの右側にあるプログラミング言語の名前を選択します。これにより、ドロップダウンが展開されます。ドロップダウンから、プログラミング言語の構文を選択します。Javascript、Python、CSS、JSON、HTML、Markdown、YAMLから選択できます。
  </TabItem>
  <TabItem value="sdk">

`wr.CodeBlock` クラスを使用して、コードブロックをプログラムで作成します。言語とコードのパラメータには、それぞれ表示したい言語名とコードを指定します。

たとえば、次の例はYAMLファイルのリストを示しています：

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(
    project='report-editing'
    )

report.blocks = [
	wr.CodeBlock(
		code=["this:", "- is", "- a", "cool:", "- yaml", "- file"],
		language="yaml"
	)
]

report.save()
```

これは、次のようなコードブロックを表示します。

```yaml
this:
- is
- a
cool:
- yaml
- file
```

以下の例は、Pythonのコードブロックを示しています。

```python
report = wr.Report(
    project='report-editing'
    )

report.blocks = [
	wr.CodeBlock(
		code = ['Hello, World!'],
		language='python'
	)
]

report.save()
```

これによって、次のようなコードブロックがレンダリングされます:

```python
Hello, World!
```
  </TabItem>
</Tabs>

### Markdown

App UIまたはWeights & Biases SDKを使って、レポートにインタラクティブにマークダウンを追加できます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">
レポートでフォワードスラッシュ (`/`) を入力すると、ドロップダウンメニューが表示されます。ドロップダウンから**Markdown**を選択してください。
  </TabItem>
  <TabItem value="sdk">

`wandb.apis.reports.MarkdownBlock` クラスを使用して、プログラムでマークダウンブロックを作成します。`text`パラメータに文字列を渡してください：

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(
    project='report-editing'
    )

report.blocks = [
	wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

これにより、以下のようなマークダウンブロックがレンダリングされます。

![](/images/reports/markdown.png)
  </TabItem>
</Tabs>

### HTML要素

App UIまたはWeights & Biases SDKを使って、レポートに対話式でHTML要素を追加してください。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">
レポートでスラッシュ（`/`）を入力することで、ドロップダウンメニューが表示されます。ドロップダウンからテキストブロックのタイプを選択してください。たとえば、H2見出しブロックを作成するには、`Heading 2`オプションを選択します。
  </TabItem>
  <TabItem value="sdk">

`wandb.apis.reports.blocks`属性に1つ以上のHTML要素のリストを渡します。次の例では、H1、H2、および順序なしリストを作成する方法を示しています。

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(
	project='report-editing'
	)

report.blocks = [
	wr.H1(text="プログラムによるレポートの仕組み"),
	wr.H2(text="見出し2"),
	wr.UnorderedList(items=["箇条書き1", "箇条書き2"])
]

report.save()
```

これにより、以下のようなHTML要素がレンダリングされます。

![](/images/reports/render_html.png)

  </TabItem>
</Tabs>
### リッチメディアリンクの埋め込み

アプリUIやWeights & Biases SDKを使用して、レポートにリッチメディアを埋め込みます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

レポートにリッチメディアを埋め込むために、URLをコピーして貼り付けます。以下のアニメーションは、Twitter、YouTube、SoundCloudからURLをコピーして貼り付ける方法を示しています。

#### Twitter

ツイートのリンクURLをコピーしてレポートに貼り付けると、レポート内でツイートが表示されます。

![](/images/reports/twitter.gif)

####

#### Youtube

YouTube動画のURLリンクをコピーして貼り付けることで、レポートに動画を埋め込むことができます。

![](/images/reports/youtube.gif)

#### SoundCloud
レポートにオーディオファイルを埋め込むには、SoundCloudのリンクをコピーして貼り付けてください。

![](/images/reports/soundcloud.gif)
  </TabItem>
  <TabItem value="sdk">

`wandb.apis.reports.blocks`属性に1つ以上の埋め込みメディアオブジェクトのリストを渡します。以下の例では、レポートにビデオとTwitterのメディアを埋め込む方法を示しています。

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(
    project='report-editing'
    )

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc%5Etfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>\n'
    )
]
report.save()
```
  </TabItem>
</Tabs>

### パネルグリッドの複製と削除

再利用したいレイアウトがある場合は、パネルグリッドを選択してコピー・ペーストして、同じレポート内で複製するか、別のレポートに貼り付けることができます。
上部右隅のドラッグハンドルを選択して、パネルグリッドの全体を強調表示します。パネルグリッド、テキスト、見出しの領域を報告書で強調表示して選択するには、クリックしてドラッグします。

![](/images/reports/demo_copy_and_paste_a_panel_grid_section.gif)

パネルグリッドを選択し、キーボードの `delete` キーを押してパネルグリッドを削除します。

![](@site/static/images/reports/delete_panel_grid.gif)

### ヘッダーを折りたたんでレポートを整理する

レポートのヘッダーを折りたたんでテキストブロック内のコンテンツを隠します。レポートが読み込まれると、展開されているヘッダーのみがコンテンツを表示します。レポートのヘッダーを折りたたむことで、コンテンツを整理し、データの読み込みを過度に行わないようにすることができます。次のgifは、そのプロセスを示しています。

![](@site/static/images/reports/collapse_headers.gif)